"""Chorus's gain on held-out windows with and without verbatim overlap with the training split, and
window-bootstrap intervals for the gains and the multipliers.

Inputs: the overlap file of `heldout_overlap.py` (.npz: per scored window, the share of its 13- and
50-token n-grams found in the training split) and, per backbone size, the per-window held-out losses
(`eval_final.full.windows.npy`) of the plain model and of the Chorus model at their best measured
rates. Both come from the same 8,192 windows, so window i is the same text in every file.

Outputs (JSON): per size, the gain over all windows; the gain on the windows whose share of 13-grams
found is 0, at most 10%, above 10% and above 50%, and on the windows with no 50-gram found; a paired
bootstrap interval (2.5% and 97.5%, 2,000 resamples of windows) for the gain; and the multipliers
read off the straight lines through the plain models, with their bootstrap interval from resampling
the windows of all models together.

    python scripts/overlap_gain.py --overlap runs/heldout_overlap/heldout_overlap_2026-11-24.npz \
        --windows runs/heldout_overlap/windows --pair 47M:screen_b_lr16_s1337:screen_e_chorus_lr8_s1337:47105088 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def equivalent(loss: float, line: list[tuple[float, float]]) -> tuple[float, bool]:
    pts = sorted(line)
    for (n0, l0), (n1, l1) in zip(pts, pts[1:]):
        if l1 <= loss <= l0:
            f = (l0 - loss) / (l0 - l1)
            return float(np.exp(np.log(n0) + f * (np.log(n1) - np.log(n0)))), False
    (n0, l0), (n1, l1) = (pts[-2], pts[-1]) if loss < pts[-1][1] else (pts[0], pts[1])
    f = (l0 - loss) / (l0 - l1)
    return float(np.exp(np.log(n0) + f * (np.log(n1) - np.log(n0)))), True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--pair", nargs="+", required=True,
                    help="size:plain_runs:chorus_runs:plain_params, runs joined by + are averaged; an empty "
                         "chorus_runs gives a plain model that only anchors the line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    ov = np.load(a.overlap)
    s13, s50 = ov["share_13"], ov["share_50"] if "share_50" in ov else None
    pairs = []
    for spec in a.pair:
        size, pr, cr, n = spec.split(":")
        # Several runs (seeds) joined by "+" are averaged window by window, as the paper averages
        # the held-out loss over seeds.
        pl = np.mean([np.load(Path(a.windows) / f"{r}.npy") for r in pr.split("+")], axis=0)
        ch = np.mean([np.load(Path(a.windows) / f"{r}.npy") for r in cr.split("+")], axis=0) if cr else None
        pairs.append((size, pr, cr, int(n), pl, ch))
    anchors = pairs
    pairs = [x for x in pairs if x[5] is not None]
    nwin = len(s13)
    bins = {"all": np.ones(nwin, bool), "13gram_none": s13 == 0, "13gram_le_10pct": s13 <= 0.10,
            "13gram_over_10pct": s13 > 0.10, "13gram_over_50pct": s13 > 0.50}
    if s50 is not None:
        bins["50gram_none"] = s50 == 0
        bins["50gram_any"] = s50 > 0
    out = {"windows": int(nwin), "bins": {k: int(v.sum()) for k, v in bins.items()}, "sizes": {}}
    rng = np.random.default_rng(1337)
    idx = rng.integers(0, nwin, size=(a.boot, nwin))
    for size, pr, cr, n, pl, ch in pairs:
        g = pl - ch
        entry = {"plain": pr, "chorus": cr, "plain_loss": round(float(pl.mean()), 5),
                 "chorus_loss": round(float(ch.mean()), 5), "gain": {}}
        for k, m in bins.items():
            if m.sum():
                entry["gain"][k] = round(float(g[m].mean()), 5)
        boot = g[idx].mean(axis=1)
        entry["gain_ci95"] = [round(float(np.percentile(boot, 2.5)), 5), round(float(np.percentile(boot, 97.5)), 5)]
        out["sizes"][size] = entry
    # Multipliers, read on the line through every plain model given, and their window-bootstrap intervals.
    line = [(n, float(pl.mean())) for _, _, _, n, pl, _ in anchors]
    for size, _, _, n, pl, ch in pairs:
        eq, ext = equivalent(float(ch.mean()), line)
        out["sizes"][size]["multiplier"] = round(eq / n, 3)
        out["sizes"][size]["extrapolated"] = ext
    mults = {size: [] for size, *_ in pairs}
    for b in range(a.boot):
        sel = idx[b]
        bl = [(n, float(pl[sel].mean())) for _, _, _, n, pl, _ in anchors]
        for size, _, _, n, pl, ch in pairs:
            eq, _ = equivalent(float(ch[sel].mean()), bl)
            mults[size].append(eq / n)
    for size, v in mults.items():
        out["sizes"][size]["multiplier_ci95"] = [round(float(np.percentile(v, 2.5)), 3),
                                                 round(float(np.percentile(v, 97.5)), 3)]
    # The multipliers read on a subset of windows only (plain line and Chorus loss on the same windows).
    for k in ("50gram_none", "13gram_none"):
        if k not in bins:
            continue
        m = bins[k]
        sub = [(n, float(pl[m].mean())) for _, _, _, n, pl, _ in anchors]
        for size, _, _, n, pl, ch in pairs:
            eq, ext = equivalent(float(ch[m].mean()), sub)
            out["sizes"][size][f"multiplier_{k}"] = round(eq / n, 3)
            out["sizes"][size][f"extrapolated_{k}"] = ext
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
