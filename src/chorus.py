"""Chorus: one set of N-gram tables read at the input and in the attention values.

After Chung and Du, "Chorus: Shared N-gram Embeddings with Layer-wise Adaptation"
(arXiv 2611.23386 v2) and their code (`reference/chorus/chorus/memory_modules.py`,
`ChorusMemory`). For the token at position t:

  1. read one row from each N-gram table (two tables per order, each of width d/4,
     addressed by the exact N-gram key in base V+1 modulo its row count) and the token
     table U [V, d]; e_t = concat(N-gram rows) + U[x_t],
  2. gate it from a fixed mix of the last three token embeddings (gradient stopped):
     [a; b] = 2 sigmoid(W_c c_t), u_t = a * e_t, z_t = u_t + SiLU(conv4(norm(u))_t),
  3. add b * W_in z_t to the token embedding (the input of block 0),
  4. in every other block (the last included), add to the attention values
     g * P_l(s_l * z_t), with s_l = 2 sigmoid(A_l norm(h)) from the residual stream
     entering the block and a per-key/value-head gate g = 2 sigmoid(G_l n[:32]) from the
     block's normalised attention input n.

Where this departs from the authors' code:

  - The backbone has no norm on its input, so step 3 adds to the embedding without one.
  - The tables are trained by the same SparseAdam as Engram's table here
    (`train_engram.py`), not by Ember, so the comparison with `e` changes the module only.
  - `orders: []` keeps the token table alone: the authors' "token table only" ablation,
    which is value embeddings in the style of nanochat and modded-nanogpt with Chorus's
    sharing and adaptation.

At initialisation the tables are zero, so the model is the backbone's function, as with
Engram's zero value projection.

Two variants of ours (none in the authors' code; the defaults give the module above
unchanged):

  - `mem_dim`: the N-gram rows concatenate to `mem_dim` values instead of d, and a linear
    map takes them to d. At the same parameters this gives d / mem_dim times as many
    rows, narrower, as Engram has (3.2M rows of 16 against Chorus's 75k of 144).
  - `keys: "word"`: the tables are addressed by words instead of tokens. Order 2 is the
    previous word with the current word up to the current token, order 3 adds the word
    before that. A word starts at a token that begins with white space or has no letters,
    and after a token with no letters (`word_map`, built by
    `scripts/build_word_map.py`); past `word_cap` tokens a word is cut after every
    token, so every key depends on a bounded number of past tokens. At
    8k tokens a token 3-gram spans about 1.3 words, so token keys mostly sit inside a word.

A regularisation of ours, also off by default: `read_dropout` p zeroes the summed table read
(token and N-gram rows) at a fraction p of the positions in training, the rest scaled by
1/(1-p); evaluation reads the whole table.

The settings live in the config's `engram` section with `"kind": "chorus"`, so the
training loop, its table optimiser and the evaluation need no change.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from engram import EngramState
from model import GPT, apply_rope, repeat_kv

DEFAULTS = {
    "kind": "chorus",
    "orders": [2, 3],
    "tables_per_order": 2,
    "rows": 100_000,
    "lr_mult": 5.0,
    "table_optimizer": "sparse_adam",
    "table_lr": None,
    "fold": None,
    "mem_dim": None,
    "keys": "token",
    "word_map": None,
    "word_cap": 8,
    "class_map": None,
    "read_dropout": 0.0,
}

# Polynomial hash of a word's tokens: below 2^31, so products of two stay inside int64.
HASH_P = 2_147_483_629
HASH_B = 1_000_003
HASH_MIX = 1_000_000_007
FROM_CHECKPOINT = "checkpoint"


def chorus_cfg(cfg: dict) -> dict:
    unknown = set(cfg["engram"]) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown chorus settings: {sorted(unknown)}")
    e = {**DEFAULTS, **cfg["engram"]}
    if e["fold"]:
        raise ValueError("chorus does not fold token ids")
    if e["keys"] not in ("token", "word", "class"):
        raise ValueError(f"chorus keys: {e['keys']}")
    if e["keys"] in ("word", "class") and (not e["word_map"] or max(e["orders"], default=2) > 3):
        raise ValueError("word and class keys need a word_map and orders within {2, 3}")
    if (e["keys"] == "class") != bool(e["class_map"]):
        raise ValueError("class keys need a class_map, and only they take one")
    return e


def value_sites(n_layer: int) -> list[int]:
    return [i for i in range(n_layer) if i % 2 == (n_layer - 1) % 2]


def _norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x.float(), (x.size(-1),), eps=1e-5)


def _previous(x: torch.Tensor, lag: int, pad: int | float = 0) -> torch.Tensor:
    """x shifted right by `lag` positions along time (dim 1), padded on the left."""
    if x.dim() == 2:
        return F.pad(x, (lag, 0), value=pad)[:, : x.size(1)]
    return F.pad(x, (0, 0, lag, 0), value=pad)[:, : x.size(1)]


class Chorus(nn.Module):
    def __init__(self, cfg: dict, generator: torch.Generator):
        super().__init__()
        e = chorus_cfg(cfg)
        d = cfg["n_embd"]
        if d % 4 or d < 32:
            raise ValueError("chorus needs a width divisible by 4 and at least 32")
        self.vocab = cfg["vocab_size"]
        self.orders = list(e["orders"])
        self.per_order = e["tables_per_order"]
        n_tables = len(self.orders) * self.per_order
        self.n_tables = n_tables
        self.mods = [e["rows"] + 2 * j for j in range(n_tables)]
        self.mem_dim = e["mem_dim"] or d
        if not 0 <= e["read_dropout"] < 1:
            raise ValueError("read_dropout must be in [0, 1)")
        self.read_dropout = e["read_dropout"]
        self.drop_seed = cfg.get("seed", 1337) + 3
        self.drop_gen = None
        self.word_keys = e["keys"] in ("word", "class")
        self.class_keys = e["keys"] == "class"
        if self.orders:
            if self.mem_dim % n_tables or d % (self.mem_dim // n_tables):
                raise ValueError("width must divide among the N-gram tables and into d")
            if not self.word_keys and (self.vocab + 1) ** max(self.orders) >= 2**62:
                raise ValueError("N-gram keys would overflow int64")
        # Every table lives in one flat embedding, read once. The N-gram tables come
        # first; then the token table U, each token's row of width d split
        # into `chunks` consecutive rows of the flat table.
        self.width = self.mem_dim // n_tables if n_tables else d // 4
        self.chunks = d // self.width
        self.offsets = [sum(self.mods[:j]) for j in range(n_tables)]
        self.token_offset = sum(self.mods)
        self.table = nn.Embedding(self.token_offset + self.vocab * self.chunks, self.width, sparse=True)
        self.expand = nn.Linear(self.mem_dim, d, bias=False) if self.mem_dim != d and n_tables else None

        self.cap = e["word_cap"]
        # Tokens of left context the keys can depend on: two for token keys (up to
        # 3-grams and the context mix); three capped words and their boundaries for words.
        self.history = 4 * self.cap + 3 if self.word_keys else 2
        if self.word_keys:
            starts = torch.zeros(2, self.vocab + 1, dtype=torch.bool)
            if e["word_map"] != FROM_CHECKPOINT:
                starts[:, : self.vocab] = torch.from_numpy(np.load(e["word_map"]).astype(bool))
            # The padding id stands before the first token: a boundary with no letters.
            starts[:, self.vocab] = True
            self.register_buffer("word_start", starts)
            n = cfg["block_size"] + 2 * self.history + 1
            self.register_buffer("hash_pow", torch.tensor(
                [pow(HASH_B, j, HASH_P) for j in range(n)], dtype=torch.long), persistent=False)
            self.register_buffer("hash_inv", torch.tensor(
                [pow(HASH_B, -j, HASH_P) for j in range(n)], dtype=torch.long), persistent=False)
        if self.class_keys:
            # Word classes of `src/class_keys_probe.py`: sorted hashes of folded pieces, their
            # classes, the fold map. A piece outside the map takes the class of its last folded
            # token (k + token), a padding piece class k + vocab. From a checkpoint the
            # buffers take the saved shapes (_load_from_state_dict).
            if e["class_map"] == FROM_CHECKPOINT:
                hashes, classes = torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long)
                fold, k = torch.arange(self.vocab + 1), 0
            else:
                m = np.load(e["class_map"])
                hashes = torch.from_numpy(m["hashes"].astype(np.int64))
                classes = torch.from_numpy(m["classes"].astype(np.int64))
                fold = torch.cat([torch.from_numpy(m["fold"].astype(np.int64)),
                                  torch.tensor([self.vocab])])
                k = int(m["k"])
            self.register_buffer("class_hash", hashes)
            self.register_buffer("class_id", classes)
            self.register_buffer("class_fold", fold)
            self.register_buffer("class_k", torch.tensor(k, dtype=torch.long))

        head_dim = d // cfg["n_head"]
        self.kv_heads = cfg["n_kv_head"]
        kv = self.kv_heads * head_dim
        self.sites = value_sites(cfg["n_layer"])
        self.context = nn.Linear(d, 2 * d, bias=False)
        self.conv = nn.Conv1d(d, d, 4, groups=d, bias=False)
        self.inject = nn.Linear(d, d, bias=False)
        self.channel = nn.ModuleDict({str(i): nn.Linear(d, d, bias=False) for i in self.sites})
        self.project = nn.ModuleDict({str(i): nn.Linear(d, kv, bias=False) for i in self.sites})
        self.head_gate = nn.ModuleDict({str(i): nn.Linear(32, self.kv_heads, bias=False)
                                        for i in self.sites})

        with torch.no_grad():
            self.table.weight.zero_()
            s = 0.25 * math.sqrt(3 / d)
            self.context.weight.copy_(torch.rand(self.context.weight.shape, generator=generator) * 2 * s - s)
            self.conv.weight.zero_()
            nn.init.eye_(self.inject.weight)
            for p in self.project.values():
                nn.init.eye_(p.weight)
                p.weight.mul_(0.25)
            for m in [*self.channel.values(), *self.head_gate.values()]:
                m.weight.zero_()
            if self.expand is not None:
                self.expand.weight.copy_(torch.randn(self.expand.weight.shape, generator=generator)
                                         / math.sqrt(self.mem_dim))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self.class_keys:
            for name in ("class_hash", "class_id", "class_fold"):
                if prefix + name in state_dict:
                    setattr(self, name, torch.empty_like(state_dict[prefix + name]))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def keys(self, idx: torch.Tensor) -> list[torch.Tensor]:
        if self.word_keys:
            return self.word_key_rows(idx)
        ids, base, out, col = idx.long(), self.vocab + 1, [], 0
        reach = 1
        for n in self.orders:
            while reach < n:
                ids = ids + _previous(idx, reach, self.vocab).long() * base**reach
                reach += 1
            for _ in range(self.per_order):
                out.append(ids.remainder(self.mods[col]) + self.offsets[col])
                col += 1
        return out

    def word_starts(self, idx: torch.Tensor) -> torch.Tensor:
        """Word piece starts [b, t]: a boundary token, a token after one with no letters,
        or, inside a word already `cap` tokens long, every further token. The rule looks
        back at most `cap` tokens, so a piece never depends on tokens far behind it."""
        x = idx.long()
        start = self.word_start[0][x] | self.word_start[1][_previous(x, 1, self.vocab)]
        pos = torch.arange(x.size(1), device=x.device).expand_as(x)
        last = torch.cummax(torch.where(start, pos, -1), dim=1).values
        return start | (pos - last >= self.cap)

    def word_key_rows(self, idx: torch.Tensor) -> list[torch.Tensor]:
        """Rows read for word keys. The sequence is padded on the left by `history`
        padding ids (boundaries), so a key at the start of a sequence equals the key a
        cached decoder computes after its padded state."""
        h = self.history
        x = F.pad(idx.long(), (h, 0), value=self.vocab)
        b, t = x.shape
        pos = torch.arange(t, device=x.device)
        start = self.word_starts(x)
        first = torch.cummax(torch.where(start, pos, 0), dim=1).values  # current piece start
        q = (x + 1) * self.hash_inv[:t] % HASH_P
        csum = F.pad(torch.cumsum(q, dim=1), (1, 0))

        def span(lo, hi, sums=csum):
            """Hash of tokens lo..hi (inclusive) per position; empty spans give 0."""
            lo_c = lo.clamp(min=0)
            h = (sums.gather(1, hi + 1) - sums.gather(1, lo_c)) % HASH_P
            h = h * self.hash_pow[hi.clamp(min=0)] % HASH_P
            return torch.where(hi >= lo_c, h, torch.zeros_like(h))

        def back(starts_at):
            """The piece before the one starting at `starts_at`: (its start, its end)."""
            end = starts_at - 1
            lo = first.gather(1, end.clamp(min=0))
            return torch.where(end >= 0, lo, end + 1), end

        hi_now = pos.expand(b, t)
        cur = span(first, hi_now)
        p0, p1 = back(first)
        q0, q1 = back(p0)
        if self.class_keys:
            xf = self.class_fold[x]
            fsum = F.pad(torch.cumsum((xf + 1) * self.hash_inv[:t] % HASH_P, dim=1), (1, 0))
            k = self.class_k
            n = k + self.vocab + 1

            def piece_class(lo, hi):
                h = span(lo, hi, fsum)
                i = torch.searchsorted(self.class_hash, h).clamp(max=self.class_hash.numel() - 1)
                found = self.class_hash[i] == h
                last = xf.gather(1, hi.clamp(min=0))
                cls = torch.where(found, self.class_id[i], k + last)
                return torch.where(hi >= 0, cls, k + self.vocab)

            c1 = piece_class(p0, p1)
            keys = {2: c1 * HASH_P + cur}
            if 3 in self.orders:
                keys[3] = (piece_class(q0, q1) * n + c1) * HASH_P + cur
        else:
            prev = span(p0, p1)
            keys = {2: prev * HASH_MIX % HASH_P * HASH_P + cur}
            if 3 in self.orders:
                keys[3] = (span(q0, q1) * HASH_MIX + prev) % HASH_P * HASH_MIX + cur
        out, col = [], 0
        for n in self.orders:
            for _ in range(self.per_order):
                out.append(keys[n][:, h:].remainder(self.mods[col]) + self.offsets[col])
                col += 1
        return out

    def forward(self, idx: torch.Tensor, word: torch.Tensor, state: EngramState | None = None,
                embed: nn.Embedding | None = None):
        """The refined vector z [b, t, d] and the input injection b * W_in z.

        With `state` (cached decoding) `idx` continues the tokens seen in earlier calls:
        the state holds the last two tokens, for the N-gram keys and the context mix, and
        the convolution's left context. `embed` is then the backbone's token embedding,
        to embed those earlier tokens again."""
        t = idx.size(1)
        if state is None:
            keys = self.keys(idx)
            w0 = _norm(word.detach())
            w1 = _previous(torch.roll(w0, 1, -1), 1)
            w2 = _previous(torch.roll(w0, 2, -1), 2)
        else:
            n = self.history
            left = state.tokens
            if left is None:
                left = idx.new_full((idx.size(0), n), self.vocab)
            padded = torch.cat([left, idx], dim=1)
            state.tokens = padded[:, -n:]
            keys = [k[:, n:] for k in self.keys(padded)]
            real = (padded != self.vocab)[..., None]
            w = _norm(embed(padded.clamp(max=self.vocab - 1)).detach()) * real
            w0 = w[:, n:]
            w1 = torch.roll(w, 1, -1)[:, n - 1 : n - 1 + t]
            w2 = torch.roll(w, 2, -1)[:, n - 2 : n - 2 + t]
        token = self.token_offset + idx.long()[..., None] * self.chunks + torch.arange(
            self.chunks, device=idx.device)
        rows = torch.cat([token, *(k[..., None] for k in keys)], dim=-1)
        b = rows.size(0)
        read = _read(self.table, rows)
        if self.expand is None:
            e = read.view(b, t, 2 if self.orders else 1, -1).sum(2)
        else:
            e = (read[:, :, : self.chunks].reshape(b, t, -1)
                 + self.expand(read[:, :, self.chunks :].reshape(b, t, -1)))
        if self.training and self.read_dropout:
            e = self._drop_positions(e)
        c = (w0 + 0.70710678 * w1 + 0.5 * w2) * 0.57735027
        pre, post = (2 * torch.sigmoid(self.context(c.to(word.dtype)))).chunk(2, dim=-1)
        u = e * pre
        x = _norm(u).to(u.dtype).transpose(1, 2)
        tail = None if state is None else state.conv_tail
        if tail is None:
            tail = x.new_zeros(b, x.size(1), 3)
        x = torch.cat([tail.to(x.dtype), x], dim=2)
        if state is not None:
            state.conv_tail = x[:, :, -3:]
        z = u + F.silu(self.conv(x)).transpose(1, 2)
        return z, self.inject(z) * post

    @torch.compiler.disable
    def _drop_positions(self, x: torch.Tensor) -> torch.Tensor:
        """Zeroes the whole read at a fraction `read_dropout` of the positions of x [b, t, d]
        and scales the rest by 1/(1-p). The mask comes from the module's own generator, so the
        global RNG and the batches stay those of the run without dropout; the generator is not
        saved with checkpoints, so a resumed run draws other masks."""
        if self.drop_gen is None or self.drop_gen.device != x.device:
            self.drop_gen = torch.Generator(device=x.device).manual_seed(self.drop_seed)
        p = self.read_dropout
        keep = torch.rand(x.shape[:2], generator=self.drop_gen, device=x.device) >= p
        return x * (keep[..., None].to(x.dtype) / (1 - p))

    def value(self, site: int, h: torch.Tensor, n: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """What block `site` adds to its attention values: [b, kv_heads, t, head_dim]."""
        key = str(site)
        s = 2 * torch.sigmoid(self.channel[key](_norm(h).to(h.dtype)))
        read = self.project[key](z * s)
        g = 2 * torch.sigmoid(self.head_gate[key](n[..., :32]))
        b, t, _ = read.shape
        read = read.view(b, t, self.kv_heads, -1) * g[..., None]
        return read.transpose(1, 2)


@torch.compiler.disable
def _read(table: nn.Embedding, rows: torch.Tensor) -> torch.Tensor:
    """The table read, kept out of torch.compile: inductor (torch 2.13) fails on the
    backward pass of a sparse embedding whose output feeds several later reads (here the
    input and every value site), with "Cannot access storage of SparseTensorImpl".
    Engram's single read compiles."""
    return table(rows)


def _block(block, x, cos, sin, extra_v, cache=None):
    """The backbone's `Block.forward` and `Attention.forward`, with `extra_v` added to the
    values before attention (and so before they enter the cache)."""
    a = block.attn
    n = block.pre_attn(x)
    b, t, _ = n.shape
    q = a.q_proj(n).view(b, t, a.n_head, a.head_dim).transpose(1, 2)
    k = a.k_proj(n).view(b, t, a.n_kv_head, a.head_dim).transpose(1, 2)
    v = a.v_proj(n).view(b, t, a.n_kv_head, a.head_dim).transpose(1, 2)
    if extra_v is not None:
        v = v + extra_v(n).to(v.dtype)
    q, k = a.q_norm(q), a.k_norm(k)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    if cache is not None:
        k, v = cache.append(k, v)
    k = repeat_kv(k, a.n_head // a.n_kv_head)
    v = repeat_kv(v, a.n_head // a.n_kv_head)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=q.size(2) == k.size(2),
                                         dropout_p=a.dropout if a.training else 0.0)
    x = x + block.post_attn(a.o_proj(out.transpose(1, 2).contiguous().view(b, t, -1)))
    return x + block.post_mlp(block.mlp(block.pre_mlp(x)))


class ChorusGPT(GPT):
    """The backbone with one Chorus module, read at the input and in the value of every
    other block."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # Built after the backbone from its own generator, and the global RNG put back, so
        # the backbone's weights and the batches are those of the run without the module.
        rng_state = torch.get_rng_state()
        gen = torch.Generator().manual_seed(cfg.get("seed", 1337) + 1)
        self.chorus = Chorus(cfg, gen)
        torch.set_rng_state(rng_state)

    def forward(self, idx, targets=None, caches=None, start_pos: int = 0):
        """With `caches` (one KVCache per block), as the backbone's cached decoding; the
        module's own state rides on the first block's cache."""
        t = idx.size(1)
        cos = self.rope_cos[start_pos : start_pos + t].to(idx.device)
        sin = self.rope_sin[start_pos : start_pos + t].to(idx.device)
        word = self.tok_emb(idx)
        state = None
        if caches is not None:
            if not hasattr(caches[0], "engram"):
                caches[0].engram = EngramState()
            state = caches[0].engram
        z, injection = self.chorus(idx, word, state, self.tok_emb)
        x = word + injection.to(word.dtype)
        sites = set(self.chorus.sites)
        for i, block in enumerate(self.blocks):
            extra = None
            if i in sites:
                h = x
                extra = lambda n, i=i, h=h: self.chorus.value(i, h, n, z.to(n.dtype))
            x = _block(block, x, cos, sin, extra, caches[i] if caches is not None else None)
        logits = self.head(self.norm_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
