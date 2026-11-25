"""Bits per byte of a checkpoint over a whole token stream, to compare models with different
vocabularies on the same text (losses per token are not comparable across tokenizers). The
stream is cut into consecutive windows of block_size inputs; every token after the first is
predicted once, from the tokens before it in its window (a larger vocabulary puts more text
into a window, which is part of what it offers). The summed cross-entropy (fp32, no
autocast) is divided by ln 2 and by the UTF-8 bytes of the predicted tokens, taken from the
byte-level vocabulary (one character of a token string is one byte; <|endoftext|> counts 0).

    PYTHONPATH=<backbone>/src:src python src/eval_bpb.py --ckpt <final.pt> --data <val.bin> \
        --tokenizer <dir with vocab.json> [--boundary <first token of the second source>] --out <json>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from engram import load_model
from train import get_device


def byte_lengths(tokenizer_dir: str, vocab_size: int) -> np.ndarray:
    vocab = json.loads((Path(tokenizer_dir) / "vocab.json").read_text(encoding="utf-8"))
    lengths = np.zeros(vocab_size, dtype=np.int64)
    for token, i in vocab.items():
        if i < vocab_size and token != "<|endoftext|>":
            lengths[i] = len(token)
    return lengths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--boundary", type=int, default=None,
                    help="token index where the second source of the stream starts")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-windows", type=int, default=None, help="for tests")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = get_device()
    model, cfg, step, n_params = load_model(args.ckpt, device)
    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    block = cfg["block_size"]
    lengths = torch.from_numpy(byte_lengths(args.tokenizer, cfg["vocab_size"]))
    n = (len(data) - 1) // block
    if args.max_windows:
        n = min(n, args.max_windows)
    boundary = args.boundary if args.boundary is not None else len(data)
    sums = {name: {"nats": 0.0, "bytes": 0, "tokens": 0} for name in ("full", "first", "second")}

    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            starts = np.arange(i, min(i + args.batch_size, n)) * block
            x = torch.from_numpy(np.stack([data[s:s + block].astype(np.int64) for s in starts])).to(device)
            y = torch.from_numpy(np.stack([data[s + 1:s + 1 + block].astype(np.int64) for s in starts])).to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1),
                                   reduction="none").view(y.shape).cpu().double()
            pos = torch.from_numpy(starts[:, None] + 1 + np.arange(block)[None, :])
            nbytes = lengths[y.cpu()]
            for name, mask in (("full", torch.ones_like(pos, dtype=torch.bool)),
                               ("first", pos < boundary), ("second", pos >= boundary)):
                sums[name]["nats"] += float(loss[mask].sum())
                sums[name]["bytes"] += int(nbytes[mask].sum())
                sums[name]["tokens"] += int(mask.sum())

    report = {"ckpt": args.ckpt, "step": step, "parameters": n_params, "vocab_size": cfg["vocab_size"],
              "block_size": block, "data": args.data, "tokenizer": args.tokenizer,
              "boundary": args.boundary, "windows": int(n), "device": device, "dtype": "fp32",
              "by_subset": {}}
    for name, s in sums.items():
        if s["tokens"]:
            report["by_subset"][name] = {
                "tokens": s["tokens"], "bytes": s["bytes"],
                "nats_per_token": round(s["nats"] / s["tokens"], 5),
                "bits_per_byte": round(s["nats"] / math.log(2) / s["bytes"], 5),
                "bytes_per_token": round(s["bytes"] / s["tokens"], 4)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report["by_subset"], indent=1))


if __name__ == "__main__":
    main()
