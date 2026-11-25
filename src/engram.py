"""Engram: a hashed N-gram embedding table read at one layer of the backbone.

After Cheng et al., "Conditional Memory via Scalable Lookup" (arXiv 2601.07372) and
DeepSeek's demo (`reference/engram_demo_v1.py`). For the token at position t:

  1. take the N-grams ending at t, for each order in `orders`,
  2. hash each with `heads` multiplicative-XOR hashes, every head into its own table
     whose size is a distinct prime,
  3. concatenate the rows read into one memory vector e_t,
  4. gate it with the hidden state: a = sigmoid(f(norm(h_t) . norm(W_K e_t) / sqrt(d))),
     v_t = a * W_V e_t, where f(x) = sign(x) sqrt(|x|) as in the demo,
  5. y_t = v_t + SiLU(causal depthwise conv(norm(v))_t), and h_t <- h_t + y_t.

Where this departs from the paper:

  - W_V starts at zero, so the model starts as exactly the function of the backbone
    without the module. In the paper only the convolution starts at zero, which leaves
    v_t in the residual stream from the first step. A baseline and its Engram arm
    sharing a seed then also share their starting point.
  - No tokenizer compression by default. The demo's map strips accents, which merges
    distinct Polish words. `fold` takes an explicit id map instead.
  - One branch: the backbone has no multi-branch residual stream (mHC).

`keys` selects what the table is addressed by. "ngram" is the method. "unigram" hashes
the current token alone into the same tables and "random" draws slots at random; both
keep every parameter and operation of the module and remove the N-gram information, so
they separate what the memory contributes from what the added parameters contribute.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from model import GPT, RMSNorm

DEFAULTS = {
    "layers": [1],
    "orders": [2, 3, 4, 5],
    "heads": 4,
    "slots": 100_003,
    "slot_dim": 32,
    "kernel": 4,
    "lr_mult": 5.0,
    "keys": "ngram",
    "fold": None,
    "hash_seed": 0,
    "table_optimizer": "sparse_adam",
    "table_lr": None,
}


def engram_cfg(cfg: dict) -> dict:
    unknown = set(cfg["engram"]) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown engram settings: {sorted(unknown)}")
    return {**DEFAULTS, **cfg["engram"]}


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def distinct_primes(start: int, count: int) -> list[int]:
    primes, n = [], start
    while len(primes) < count:
        if is_prime(n):
            primes.append(n)
        n += 1
    return primes


class EngramState:
    """What cached decoding has to remember: the last tokens, for N-grams that reach
    back before the current call, and the convolution's left context."""

    def __init__(self):
        self.tokens = None
        self.conv_tail = None


class Engram(nn.Module):
    def __init__(self, cfg: dict, generator: torch.Generator):
        super().__init__()
        e = engram_cfg(cfg)
        d = cfg["n_embd"]
        self.orders = list(e["orders"])
        self.heads = e["heads"]
        self.max_order = max(self.orders)
        self.keys = e["keys"]
        if self.keys not in ("ngram", "unigram", "random"):
            raise ValueError(f"engram keys: {self.keys}")
        self.pad_id = cfg["vocab_size"]
        n_tables = len(self.orders) * self.heads

        primes = distinct_primes(e["slots"], n_tables)
        self.register_buffer("mods", torch.tensor(primes, dtype=torch.long))
        self.register_buffer("offsets", torch.tensor(np.cumsum([0] + primes[:-1]), dtype=torch.long))
        # Odd and below 2^50, so a product with a token id stays inside int64.
        rng = np.random.default_rng(e["hash_seed"])
        mults = rng.integers(0, 2**49, size=(n_tables, self.max_order), dtype=np.int64) * 2 + 1
        self.register_buffer("mults", torch.from_numpy(mults))
        fold = torch.arange(cfg["vocab_size"] + 1)
        if e["fold"]:
            fold[: cfg["vocab_size"]] = torch.from_numpy(np.load(e["fold"]).astype(np.int64))
        self.register_buffer("fold", fold)

        mem_dim = n_tables * e["slot_dim"]
        self.table = nn.Embedding(sum(primes), e["slot_dim"], sparse=True)
        self.key = nn.Linear(mem_dim, d, bias=False)
        self.value = nn.Linear(mem_dim, d, bias=False)
        self.norm_h, self.norm_k, self.norm_v = RMSNorm(d), RMSNorm(d), RMSNorm(d)
        self.dilation = self.max_order
        self.left = (e["kernel"] - 1) * self.dilation
        self.conv = nn.Conv1d(d, d, e["kernel"], groups=d, dilation=self.dilation, bias=False)
        self.scale = 1.0 / math.sqrt(d)
        self._random = None

        with torch.no_grad():
            self.table.weight.copy_(torch.randn(self.table.weight.shape, generator=generator) * 0.02)
            self.key.weight.copy_(torch.randn(self.key.weight.shape, generator=generator) * 0.02)
            self.value.weight.zero_()
            self.conv.weight.zero_()

    def slots(self, idx: torch.Tensor, state: EngramState | None = None) -> torch.Tensor:
        """Rows of the flat table read at every position: [batch, time, orders * heads]."""
        b, t = idx.shape
        if self.keys == "random":
            if self._random is None or self._random.device != idx.device:
                self._random = torch.Generator(device=idx.device)
                self._random.manual_seed(0)
            u = torch.rand((b, t, len(self.mods)), device=idx.device, generator=self._random,
                           dtype=torch.float64)
            return (u * self.mods).long() % self.mods + self.offsets
        tok = self.fold[idx]
        left = None if state is None else state.tokens
        if left is None:
            left = tok.new_full((b, self.max_order - 1), self.pad_id)
        padded = torch.cat([left, tok], dim=1)
        if state is not None:
            state.tokens = padded[:, -(self.max_order - 1):]
        # back[i] is the token i positions before the current one.
        back = [padded[:, self.max_order - 1 - i : self.max_order - 1 - i + t]
                for i in range(self.max_order)]
        out, col = [], 0
        for n in self.orders:
            span = 1 if self.keys == "unigram" else n
            for _ in range(self.heads):
                mix = back[0] * self.mults[col, 0]
                for i in range(1, span):
                    mix = mix ^ (back[i] * self.mults[col, i])
                out.append(mix % self.mods[col] + self.offsets[col])
                col += 1
        return torch.stack(out, dim=-1)

    def forward(self, h: torch.Tensor, idx: torch.Tensor, state: EngramState | None = None):
        e = self.table(self.slots(idx, state)).flatten(-2)
        gate = (self.norm_h(h) * self.norm_k(self.key(e))).sum(-1, keepdim=True) * self.scale
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        v = torch.sigmoid(gate) * self.value(e)

        x = self.norm_v(v).transpose(1, 2)
        tail = None if state is None else state.conv_tail
        if tail is None:
            tail = x.new_zeros(x.size(0), x.size(1), self.left)
        x = torch.cat([tail.to(x.dtype), x], dim=2)
        if state is not None:
            state.conv_tail = x[:, :, -self.left:]
        return h + v + F.silu(self.conv(x)).transpose(1, 2)


LOOP_DEFAULTS = {"first": 3, "last": 8, "passes": 3}


def loop_cfg(cfg: dict) -> dict | None:
    """Blocks `first` to `last` (inclusive) run `passes` times with shared weights, as the
    looped middle of Inkling 2 Mini (thinkingmachines.ai/news/inkling-2-mini/)."""
    if not cfg.get("loop"):
        return None
    unknown = set(cfg["loop"]) - set(LOOP_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown loop settings: {sorted(unknown)}")
    loop = {**LOOP_DEFAULTS, **cfg["loop"]}
    if not 0 <= loop["first"] <= loop["last"] < cfg["n_layer"] or loop["passes"] < 1:
        raise ValueError(f"bad loop {loop} for {cfg['n_layer']} blocks")
    return loop


class EngramGPT(GPT):
    """The backbone with an Engram module before each block listed in `engram.layers`
    (0-based, so [1] is the second block), and optionally a looped middle (`loop`).

    With a loop, every pass after the first starts again from the input of the first pass
    and adds the previous pass's output through a per-channel gate that starts at zero:
    x_p = x_0 + g_p * norm(y_{p-1}). At the start every pass computes y_1, so the model is
    the function of the backbone without the loop, from the backbone's own weights. Each
    application of a block has its own attention cache: `caches` is indexed by
    application, and a list shorter than that is extended in place."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        e = engram_cfg(cfg) if cfg.get("engram") else {"layers": []}
        # The module is built after the backbone and from its own generator, and the
        # global RNG is put back afterwards: the backbone's weights and every later draw
        # (the training batches) are then the same as in a run without the module.
        rng_state = torch.get_rng_state()
        gen = torch.Generator().manual_seed(cfg.get("seed", 1337) + 1)
        self.engram = nn.ModuleDict({str(i): Engram(cfg, gen) for i in e["layers"]})
        torch.set_rng_state(rng_state)

        self.loop = loop_cfg(cfg)
        self.order = [(i, 0) for i in range(cfg["n_layer"])]
        if self.loop:
            first, last, passes = self.loop["first"], self.loop["last"], self.loop["passes"]
            if any(first <= i <= last for i in e["layers"]) and passes > 1:
                raise ValueError("an Engram module inside the looped blocks")
            middle = [(i, p) for p in range(passes) for i in range(first, last + 1)]
            self.order = self.order[:first] + middle + self.order[last + 1:]
            d = cfg["n_embd"]
            self.loop_norm = RMSNorm(d)
            self.loop_gate = nn.ParameterList(nn.Parameter(torch.zeros(d)) for _ in range(passes - 1))
        self.n_applications = len(self.order)

    def forward(self, idx, targets=None, caches=None, start_pos: int = 0):
        t = idx.size(1)
        cos = self.rope_cos[start_pos : start_pos + t].to(idx.device)
        sin = self.rope_sin[start_pos : start_pos + t].to(idx.device)
        if caches is not None and len(caches) < self.n_applications:
            caches.extend(type(caches[0])() for _ in range(self.n_applications - len(caches)))
        x = self.tok_emb(idx)
        start = None
        for a, (i, p) in enumerate(self.order):
            block = self.blocks[i]
            cache = caches[a] if caches is not None else None
            if self.loop and i == self.loop["first"]:
                if p == 0:
                    start = x
                else:
                    x = start + self.loop_gate[p - 1] * self.loop_norm(x)
            if str(i) in self.engram:
                state = None
                if cache is not None:
                    if not hasattr(cache, "engram"):
                        cache.engram = EngramState()
                    state = cache.engram
                x = self.engram[str(i)](x, idx, state)
            x = block(x, cos, sin, cache)
        logits = self.head(self.norm_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def build_model(cfg: dict) -> GPT:
    if cfg.get("emb"):
        if cfg.get("engram") or cfg.get("loop"):
            raise ValueError("emb is for the plain backbone only")
        from embed import EmbedGPT
        return EmbedGPT(cfg)
    if cfg.get("engram", {}).get("kind") == "chorus":
        if cfg.get("loop"):
            raise ValueError("no loop for Chorus")
        from chorus import ChorusGPT
        return ChorusGPT(cfg)
    return EngramGPT(cfg) if cfg.get("engram") or cfg.get("loop") else GPT(cfg)


def load_model(ckpt_path: str, device: str, dtype: torch.dtype = torch.float32):
    """Same contract as `eval_val.load_model`, for checkpoints with or without Engram. The file
    is memory-mapped, so the optimizer states it also holds are never read (e_chorus_huge's
    final.pt is 12.7 GB, of which 4.1 GB are weights)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    cfg = ckpt["cfg"]
    if cfg.get("engram"):
        cfg = {**cfg, "engram": {**cfg["engram"], "fold": None}}
        for name in ("word_map", "class_map"):
            if cfg["engram"].get(name):
                cfg["engram"][name] = "checkpoint"  # the map is a buffer of the checkpoint
    model = build_model(cfg)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()})
    n_params = sum(p.numel() for p in model.parameters())
    return model.to(device=device, dtype=dtype).eval(), ckpt["cfg"], int(ckpt.get("step", -1)), n_params


def is_table(name: str) -> bool:
    return ".table." in name


@torch.no_grad()
def zero_table(model: GPT, orders: list[int] | None = None) -> int:
    """Zeroes the rows of the Engram tables that hold the given N-gram orders (every
    order when None) and returns how many rows were zeroed. A row of zeros makes the
    module add nothing for that order's heads; with every order zeroed the model computes
    the backbone it was trained with, without its memory.

    For Chorus the N-gram tables of those orders are zeroed and the token rows stay (they
    are per-token values, like value embeddings), so the model keeps its token table and
    loses its N-gram memory."""
    rows = 0
    if hasattr(model, "chorus"):
        c = model.chorus
        unknown = set(orders or []) - set(c.orders)
        if unknown:
            raise ValueError(f"orders not in the module: {sorted(unknown)}")
        for i, n in enumerate(c.orders):
            if orders is not None and n not in orders:
                continue
            for col in range(i * c.per_order, (i + 1) * c.per_order):
                c.table.weight[c.offsets[col] : c.offsets[col] + c.mods[col]] = 0
                rows += c.mods[col]
        return rows
    if not isinstance(model, EngramGPT):
        raise ValueError("zero_table takes an Engram or Chorus model")
    for module in model.engram.values():
        unknown = set(orders or []) - set(module.orders)
        if unknown:
            raise ValueError(f"orders not in the module: {sorted(unknown)}")
        for i, n in enumerate(module.orders):
            if orders is not None and n not in orders:
                continue
            for col in range(i * module.heads, (i + 1) * module.heads):
                start, size = int(module.offsets[col]), int(module.mods[col])
                module.table.weight[start : start + size] = 0
                rows += size
    return rows


@torch.no_grad()
def quantize_table(model: GPT, bits: int) -> dict:
    """Rounds every Engram table row to `bits`-bit signed integers with one scale per row
    (the row's largest magnitude), and writes the rounded values back as floats, so the
    model computes what it would with the stored integers. Returns the storage it implies:
    the integers plus one 16-bit scale per row.

    For Chorus every row of its one table is rounded, the token rows included. Rows are
    processed in chunks so that a table of a billion values needs little memory beyond its
    own."""
    if hasattr(model, "chorus"):
        tables = [model.chorus.table.weight]
    elif isinstance(model, EngramGPT) and model.engram:
        tables = [module.table.weight for module in model.engram.values()]
    else:
        raise ValueError("quantize_table takes an Engram or Chorus model")
    if not 2 <= bits <= 8:
        raise ValueError(f"bits: {bits}")
    levels = 2 ** (bits - 1) - 1
    rows = values = 0
    err2 = norm2 = 0.0
    for table in tables:
        for w in table.split(1 << 20):
            scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / levels
            q = (w / scale).round().clamp(-levels, levels) * scale
            err2 += float((q - w).float().square().sum())
            norm2 += float(w.float().square().sum())
            w.copy_(q)
        rows += table.size(0)
        values += table.numel()
    return {"bits": bits, "rows": rows, "bytes": values * bits // 8 + rows * 2,
            "bytes_bf16": values * 2, "relative_error": (err2 / max(norm2, 1e-30)) ** 0.5}
