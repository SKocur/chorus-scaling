"""Verbatim overlap between the held-out windows that the evaluation scores and the training split.

The evaluation (`eval_val.evaluate` of the backbone) scores 8,192 non-overlapping windows of 1,024
tokens spread evenly across the held-out file; training draws windows at random from the whole
training split. For every scored window this script finds which of its token n-grams (n = 13 and
n = 50 by default) occur anywhere in the training split, and writes per window the share of its
n-grams found, and per token whether the n-gram ending at that token was found. A 50-token match is
a long verbatim passage (a reprint, a repeated notice); a 13-token match can also be a set phrase.

Method: the held-out n-grams are few (about 8.4 million per n), so they are hashed and kept in a
sorted array with a bitmap in front of it; the training split is streamed in chunks, each chunk's
n-grams are hashed the same way and looked up. Hashes are polynomial in 64-bit arithmetic over the
token ids; a sample of the matches is checked token by token against the training split.

    nice -n 19 python3 scripts/heldout_overlap.py --train <tokens_frozen_6.69B.bin> \
        --val <val_2026-08-03.bin> --out metrics/heldout_overlap_2026-11-24
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

BASE = np.uint64(0x9E3779B97F4A7C15)
BITS = 28


def ngram_hashes(tokens: np.ndarray, n: int) -> np.ndarray:
    """Hash of every n-gram of `tokens`; entry i covers tokens[i : i + n]."""
    x = tokens.astype(np.uint64) + np.uint64(1)
    m = len(x) - n + 1
    if m <= 0:
        return np.empty(0, dtype=np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(n):
            h = h * BASE + x[k:k + m]
    return h


def window_starts(n_tokens: int, block: int, n_windows: int) -> np.ndarray:
    """The windows `eval_val.evaluate` scores (same arithmetic)."""
    max_windows = (n_tokens - 1) // block
    n = min(n_windows, max_windows)
    return np.linspace(0, max_windows - 1, n, dtype=np.int64) * block


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True, help="output path without extension (.json and .npz)")
    ap.add_argument("--n", type=int, nargs="+", default=[13, 50])
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--windows", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=150_000_000)
    a = ap.parse_args()

    train = np.memmap(a.train, dtype=np.uint16, mode="r")
    val = np.memmap(a.val, dtype=np.uint16, mode="r")
    starts = window_starts(len(val), a.block, a.windows)
    # Each window covers the model's input and its targets: tokens s .. s + block.
    spans = [np.asarray(val[s:s + a.block + 1]) for s in starts]
    print(f"train {len(train):,} tokens, val {len(val):,}, {len(starts)} windows", flush=True)

    result = {"train": a.train, "val": a.val, "train_tokens": int(len(train)), "val_tokens": int(len(val)),
              "windows": int(len(starts)), "block": a.block, "per_n": {}}
    arrays = {"starts": starts}
    for n in a.n:
        t0 = time.time()
        per_window = [ngram_hashes(s, n) for s in spans]
        allh = np.concatenate(per_window)
        uniq = np.unique(allh)
        bitmap = np.zeros(1 << BITS, dtype=bool)
        bitmap[(uniq >> np.uint64(64 - BITS)).astype(np.int64)] = True
        found = np.zeros(len(uniq), dtype=bool)
        first_pos = np.full(len(uniq), -1, dtype=np.int64)
        step = a.chunk
        for c0 in range(0, len(train) - n + 1, step):
            c1 = min(len(train), c0 + step + n - 1)
            h = ngram_hashes(np.asarray(train[c0:c1]), n)
            cand = np.nonzero(bitmap[(h >> np.uint64(64 - BITS)).astype(np.int64)])[0]
            if len(cand):
                hc = h[cand]
                idx = np.searchsorted(uniq, hc)
                idx[idx == len(uniq)] = 0
                hit = uniq[idx] == hc
                new = hit & ~found[idx]
                first_pos[idx[new]] = c0 + cand[new]
                found[idx[hit]] = True
            print(f"  n={n} {c1 / len(train):6.1%} {found.sum():,}/{len(uniq):,} found "
                  f"{time.time() - t0:.0f} s", flush=True)
        # Check a sample of matches token by token against the training split.
        rng = np.random.default_rng(0)
        checked = ok = 0
        for w in rng.permutation(len(per_window)):
            if checked >= 200:
                break
            idx = np.searchsorted(uniq, per_window[w])
            pos = np.nonzero(found[idx])[0]
            if not len(pos):
                continue
            i = int(rng.choice(pos))
            p = int(first_pos[idx[i]])
            checked += 1
            ok += int(np.array_equal(np.asarray(train[p:p + n]), spans[w][i:i + n]))
        share = []
        token_hit = []
        for hw in per_window:
            f = found[np.searchsorted(uniq, hw)]
            share.append(f.mean() if len(f) else 0.0)
            # The n-gram ending at target t (window offset t >= n - 1) found in training.
            token_hit.append(np.concatenate([np.zeros(n - 1, dtype=bool), f]))
        share = np.array(share)
        arrays[f"share_{n}"] = share
        arrays[f"tokenhit_{n}"] = np.stack(token_hit)
        result["per_n"][str(n)] = {
            "unique_ngrams": int(len(uniq)), "found": int(found.sum()),
            "share_of_scored_ngrams_found": round(float(np.concatenate(
                [found[np.searchsorted(uniq, hw)] for hw in per_window]).mean()), 6),
            "windows_with_any": int((share > 0).sum()),
            "windows_over_10pct": int((share > 0.10).sum()),
            "windows_over_50pct": int((share > 0.50).sum()),
            "windows_over_90pct": int((share > 0.90).sum()),
            "match_check": {"checked": checked, "equal": ok},
            "seconds": round(time.time() - t0),
        }
        print(json.dumps(result["per_n"][str(n)]), flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
