"""Held-out loss of a checkpoint with or without Engram, whole split and by source.

Calls the backbone's `eval_val.evaluate`, so windows, stride and precision are the ones
behind the ladder's published numbers; only the loader differs, because it has to build
the module. The per-source numbers follow `scripts/eval_by_source.py` of the main repo:
each source is its own contiguous block of the split and is scored as its own stream.

    PYTHONPATH=<backbone>/src python src/eval_engram.py --ckpt <ckpt> --data <val.bin> \
        --windows 8192 --out runs/<run>/eval_final.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from engram import is_table, load_model, quantize_table, zero_table
from eval_val import evaluate, file_fingerprint
from train import get_device


def summary(losses: np.ndarray, meta: dict) -> dict:
    ce = float(losses.mean())
    sem = float(losses.std(ddof=1) / math.sqrt(len(losses)))
    return {"cross_entropy_nats": round(ce, 5), "sem": round(sem, 5),
            "ci95": [round(ce - 1.96 * sem, 5), round(ce + 1.96 * sem, 5)], **meta}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--windows", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-source-boundary", type=int, default=55_758_336)
    p.add_argument("--no-by-source", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--zero-table", nargs="+", default=None, metavar="ORDER",
                   help="zero the Engram table rows of these N-gram orders, or 'all'")
    p.add_argument("--table-bits", type=int, default=None,
                   help="round the Engram table to this many bits per value, one scale per row")
    args = p.parse_args()

    device = get_device()
    model, cfg, step, n_params = load_model(args.ckpt, device)
    zeroed = None
    if args.zero_table:
        zeroed = zero_table(model, None if args.zero_table == ["all"] else [int(n) for n in args.zero_table])
    quantized = quantize_table(model, args.table_bits) if args.table_bits else None
    n_table = sum(p.numel() for n, p in model.named_parameters() if is_table(n))
    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    block = cfg["block_size"]

    subsets = {"full": data}
    if not args.no_by_source:
        subsets["internet_archive"] = data[: args.val_source_boundary]
        subsets["wolne_lektury"] = data[args.val_source_boundary :]
    report = {
        "ckpt": args.ckpt, "step": step, "params_outside_table": n_params - n_table,
        "params_table": n_table, "engram": cfg.get("engram"), "device": device,
        "dtype": "fp32", "batch_size": args.batch_size, "block_size": block,
        "data": file_fingerprint(Path(args.data)), "zeroed_table_rows": zeroed,
        "quantized_table": quantized, "by_subset": {},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name, stream in subsets.items():
        losses, meta = evaluate(model, stream, block, args.windows, args.batch_size,
                                device, torch.float32)
        report["by_subset"][name] = summary(losses, meta)
        np.save(out.with_suffix(f".{name}.windows.npy"), losses)
        print(f"{name}: CE {report['by_subset'][name]['cross_entropy_nats']:.5f} "
              f"± {1.96 * report['by_subset'][name]['sem']:.5f}", flush=True)
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
