"""Training loop for a baseline and its Engram arm.

The same loop as the backbone's `train.py` (seed, batches, schedule, Muon + AdamW,
clipping, evaluation windows), so a config without an `engram` section trains the
baseline, and the two arms differ by that section alone. What the Engram arm adds:

  - the table is trained by its own Adam (sparse, no weight decay) at `lr * lr_mult`,
    following the paper, or with `engram.table_optimizer: "ember"` by Ember at
    `engram.table_lr` (`src/ember.py`); every other parameter of the module goes where the backbone's
    rule sends it (2D weights to Muon, the rest to AdamW),
  - gradient clipping covers every parameter except the table, so the backbone's
    clipped norm means the same in both arms.

An untied head (`emb.untied`, `src/embed.py`) goes to AdamW with the token table; tied, it is
the token table and appears once among the parameters.

Checkpoints and metrics go to the directories given on the command line.

    PYTHONPATH=<backbone>/src python src/train_engram.py --config configs/<c>.json \
        --data <tokens.bin> --val-data <val.bin> --ckpt-dir <dir> --metrics-dir <dir>
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ember import Ember, table_segments
from engram import build_model, is_table
from metrics_logger import MetricsLogger
from train import Muon, estimate_loss, file_sha256, get_batch, get_device, lr_mult


def build_optimizers(model, cfg: dict):
    muon_params, adamw_params, table_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_table(name):
            table_params.append(p)
        elif p.ndim == 2 and "tok_emb" not in name and name.removeprefix("_orig_mod.") != "head.weight":
            muon_params.append(p)
        else:
            adamw_params.append(p)
    muon = Muon(muon_params, lr=cfg["muon_lr"], momentum=0.95)
    adamw = torch.optim.AdamW(adamw_params, lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.0)
    table = None
    if table_params:
        if table_optimizer(cfg) == "ember":
            table = Ember(table_params, [table_segments(model, p) for p in table_params],
                          lr=table_lr(cfg))
        else:
            table = torch.optim.SparseAdam(table_params, lr=table_lr(cfg), betas=(0.9, 0.95))
    return muon, adamw, table


def table_optimizer(cfg: dict) -> str:
    return cfg["engram"].get("table_optimizer", "sparse_adam")


def table_lr(cfg: dict) -> float:
    """Base learning rate of the table: SparseAdam at `lr * lr_mult` (5 by default, after
    the Engram paper), Ember at `engram.table_lr`."""
    if table_optimizer(cfg) == "ember":
        return cfg["engram"]["table_lr"]
    return cfg["lr"] * cfg["engram"].get("lr_mult", 5.0)


def save_ckpt(path: Path, raw_model, optimizers: dict, step: int, cfg: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model": raw_model.state_dict(),
            **{name: opt.state_dict() for name, opt in optimizers.items() if opt is not None},
            "step": step,
            "cfg": cfg,
            "rng": {"torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        },
        tmp,
    )
    tmp.replace(path)
    print(f"           ckpt {path.name} @ step {step}", flush=True)


def prune_ckpts(ckpt_dir: Path, keep_last: int | None, keep_every: int | None) -> None:
    if not keep_last:
        return
    saved = sorted((int(p.stem.removeprefix("step")), p) for p in ckpt_dir.glob("step*.pt"))
    for s, p in saved[:-keep_last]:
        if keep_every and s % keep_every == 0:
            continue
        p.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stop-after", type=int, default=None,
                        help="end after this step, with the schedule of the full run (probes)")
    parser.add_argument("--resume")
    parser.add_argument("--run", default=None, help="run name; default <config>_<date>_s<seed>")
    parser.add_argument("--no-checkpoints", action="store_true", help="for timing and smoke runs")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    seed = args.seed if args.seed is not None else cfg.get("seed", 1337)
    cfg["seed"] = seed
    device = get_device()
    torch.manual_seed(seed)

    train_data = np.memmap(args.data, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode="r")
    print(f"Corpus: {len(train_data):,} train tokens ({len(val_data):,} val) | device: {device}")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_table = sum(p.numel() for n, p in model.named_parameters() if is_table(n))
    print(f"Model: {(n_params - n_table) / 1e6:.1f}M params outside the table, "
          f"{n_table / 1e6:.1f}M in the table", flush=True)

    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()})

    raw_model = model
    if cfg.get("compile") and device == "cuda":
        model = torch.compile(model)

    muon, adamw, table = build_optimizers(model, cfg)
    optimizers = {"muon": muon, "adamw": adamw, "table": table}
    schedule = [(muon, cfg["muon_lr"]), (adamw, cfg["lr"])]
    if table is not None:
        schedule.append((table, table_lr(cfg)))
    clipped = [p for n, p in model.named_parameters() if not is_table(n)]
    # Decoupled weight decay on the blocks' matrices only (`muon_weight_decay`, default 0),
    # p <- p * (1 - lr * wd) after each step at Muon's scheduled lr: the backbone pays for what
    # it holds, the table and the module's own weights do not (research-plan.md, idea 2).
    muon_wd = cfg.get("muon_weight_decay", 0.0)
    decayed = [p for n, p in model.named_parameters()
               if n.removeprefix("_orig_mod.").startswith("blocks.") and p.ndim == 2] if muon_wd else []
    grad_accum = cfg.get("grad_accum", 1)

    start_step = 0
    if ckpt is not None:
        start_step = ckpt["step"] + 1
        for name, opt in optimizers.items():
            if opt is not None:
                opt.load_state_dict(ckpt[name])
        torch.set_rng_state(ckpt["rng"]["torch"].cpu())
        if ckpt["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng"]["cuda"]])
        print(f"Resuming from {args.resume} at step {start_step:,}")

    run = args.run or f"{Path(args.config).stem}_{datetime.date.today().isoformat()}_s{seed}"
    ckpt_dir, metrics_dir = Path(args.ckpt_dir), Path(args.metrics_dir)
    if not args.no_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tokens_per_step = cfg["batch_size"] * grad_accum * cfg["block_size"]
    train_log = MetricsLogger(metrics_dir / f"{run}_train.csv",
                              ["step", "tokens", "elapsed_s", "unix_ts", "loss",
                               "grad_norm", "lr_mult", "tok_s"])
    val_log = MetricsLogger(metrics_dir / f"{run}_val.csv",
                            ["step", "tokens", "elapsed_s", "unix_ts", "val_loss"])
    src = Path(__file__).resolve().parent
    (metrics_dir / f"{run}_manifest.json").write_text(json.dumps({
        "run": run, "seed": seed, "config": args.config, "config_values": cfg,
        "params_outside_table": n_params - n_table, "params_table": n_table,
        "train_tokens": int(len(train_data)), "val_tokens": int(len(val_data)),
        "data": {"train": args.data, "train_bytes": Path(args.data).stat().st_size,
                 "val": args.val_data, "val_sha256": file_sha256(Path(args.val_data))},
        "source_sha256": {f: file_sha256(src / f) for f in ("train_engram.py", "engram.py", "ember.py")},
        "backbone_sha256": {m: file_sha256(Path(sys.modules[m].__file__))
                            for m in ("train", "model")},
        "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                "cuda": torch.version.cuda, "device": device,
                "device_name": (torch.cuda.get_device_name(0)
                                if device == "cuda" else platform.processor())},
        "resume": args.resume, "started_unix": time.time(),
    }, indent=2))

    decay_start = None
    if cfg.get("schedule", "wsd") != "cosine":
        decay_start = cfg["max_steps"] - max(1, int(cfg["max_steps"] * cfg.get("decay_frac", 0.1)))
    steps_per_epoch = max(1, len(train_data) // tokens_per_step)
    t0 = time.time()
    last_log_step, last_log_t = start_step, t0
    model.train()
    for step in range(start_step, cfg["max_steps"]):
        if args.stop_after is not None and step > args.stop_after:
            break
        mult = lr_mult(step, cfg)
        for opt, base in schedule:
            for group in opt.param_groups:
                group["lr"] = base * mult

        for opt in optimizers.values():
            if opt is not None:
                opt.zero_grad(set_to_none=True)
        loss_val = 0.0
        for _ in range(grad_accum):
            x, y = get_batch(train_data, cfg, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = model(x, y)
            (loss / grad_accum).backward()
            loss_val += loss.item() / grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(clipped, 1.0)
        for opt in optimizers.values():
            if opt is not None:
                opt.step()
        if decayed:
            with torch.no_grad():
                torch._foreach_mul_(decayed, 1 - muon.param_groups[0]["lr"] * muon_wd)

        if step % cfg["log_every"] == 0:
            now = time.time()
            tok_s = (step - last_log_step) * tokens_per_step / max(1e-9, now - last_log_t)
            last_log_step, last_log_t = step, now
            print(f"step {step:>6} | loss {loss_val:.4f} | grad_norm {grad_norm:.3f}"
                  f" | lr x{mult:.3f} | {tok_s/1e3:,.0f}k tok/s", flush=True)
            train_log.log(step=step, tokens=step * tokens_per_step, elapsed_s=f"{now - t0:.1f}",
                          unix_ts=f"{now:.1f}", loss=f"{loss_val:.4f}",
                          grad_norm=f"{grad_norm:.3f}", lr_mult=f"{mult:.4f}",
                          tok_s=f"{tok_s:.0f}")
        if step == start_step + 10 and device == "cuda":
            print(f"           VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB used"
                  f" / {torch.cuda.max_memory_reserved() / 1e9:.1f} GB reserved", flush=True)
        if step % cfg.get("eval_every", 250) == 0:
            vl = estimate_loss(model, val_data, cfg, device, cfg.get("eval_iters", 20))
            print(f"           val loss {vl:.4f}", flush=True)
            val_log.log(step=step, tokens=step * tokens_per_step,
                        elapsed_s=f"{time.time() - t0:.1f}", unix_ts=f"{time.time():.1f}",
                        val_loss=f"{vl:.4f}")
        if args.no_checkpoints:
            continue
        if decay_start is not None and step == decay_start:
            save_ckpt(ckpt_dir / "predecay.pt", raw_model, optimizers, step, cfg)
        if cfg.get("branch_ckpt_step") is not None and step == cfg["branch_ckpt_step"]:
            save_ckpt(ckpt_dir / "stable_branch.pt", raw_model, optimizers, step, cfg)
        if step > 0 and step % cfg["ckpt_every"] == 0:
            save_ckpt(ckpt_dir / f"step{step}.pt", raw_model, optimizers, step, cfg)
            prune_ckpts(ckpt_dir, cfg.get("keep_last"), cfg.get("keep_every"))
        if cfg.get("ckpt_every_epoch") and step > 0 and step % steps_per_epoch == 0:
            save_ckpt(ckpt_dir / f"epoch{step // steps_per_epoch}.pt", raw_model, optimizers, step, cfg)

    if not args.no_checkpoints:
        save_ckpt(ckpt_dir / "final.pt", raw_model, optimizers, cfg["max_steps"] - 1, cfg)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
