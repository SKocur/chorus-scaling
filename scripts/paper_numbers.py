"""Writes the preprint's result tables and numbers from the run outputs. Reads
metrics/screen/<run>/ (eval_final.json, eval_bpb.json, the manifest) and metrics/fact_quiz/; writes
paper/tables/*.tex and paper/numbers.tex. Light JSON parsing only.

Equivalent size: the loss of the models without a table (b, at the same learning-rate factor,
seeds averaged) read off the straight line between neighbouring measured sizes on a logarithmic size
axis; outside the measured sizes the nearest segment is extended and the value is marked.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCREEN = ROOT / "metrics" / "screen"
QUIZ = ROOT / "metrics" / "fact_quiz"
QUANT = ROOT / "metrics" / "quant_chorus"
OUT = ROOT / "paper"

SIZES = {"16m": "16M", "": "47M", "107m": "107M", "203m": "204M", "349m": "349M", "550m": "550M", "818m": "818M"}
ORDER = ["16M", "47M", "107M", "204M", "349M", "550M", "818M"]
TABLES = {"small": "0.275", "": "1.09", "big": "4.35", "huge": "max"}
NAME = re.compile(r"^screen_(?P<arm>b_ve|b|e_chorus)(?:_(?P<tab>small|big|huge))?(?:_(?P<size>16m|107m|203m|349m|550m|818m))?"
                  r"(?:_lr(?P<lr>0p5|2|4|8|16|32))?(?:_mb(?:8|4|2|1))?_s(?P<seed>\d+)$")
# Runs at half and twice the token budget (scripts/run_branch_rented.sh).
BUDGET_NAME = re.compile(r"^screen_(?P<arm>b|e_chorus)(?:_(?P<size>16m|107m|203m|349m|550m))?_lr(?P<lr>\d+)"
                         r"_(?P<kind>half|halffresh|double)_s(?P<seed>\d+)$")
# Side runs at 47M: the tables' own rate (lr_mult, 5 by default, so the factor is relative to the default)
# and the authors' Ember optimiser for the tables.
SIDE_RUNS = {"tablerate0p33": ["screen_e_chorus_lrmult1p67_s1337"],
             "tablerate3": ["screen_e_chorus_lrmult15_s1337", "screen_e_chorus_lrmult15_s2"],
             "tablerate9": ["screen_e_chorus_lrmult45_mb8_s1337"],
             "tablerate3-lr8": ["screen_e_chorus_lr8_tr3_s1337"],
             "tablerate0p33-lr8": ["screen_e_chorus_lr8_tr0p33_s1337"],
             "tablerate1-lr8": ["screen_e_chorus_lr8_s1337"],
             "ember": ["screen_e_chorus_ember_s1337"],
             "ember-lr0p023": ["screen_e_chorus_ember_lr0p023_s1337"],
             "ember-lr0p073": ["screen_e_chorus_ember_lr0p073_s1337"],
             "ember-lr0p73": ["screen_e_chorus_ember_lr0p73_s1337"]}
# Chorus at its best backbone rate per size with a lower tables' rate: <stem>_<tag>_s<seed>, against <stem>_s<seed>.
TR_STEMS = {"16M": "screen_e_chorus_16m_lr16", "47M": "screen_e_chorus_lr8", "107M": "screen_e_chorus_107m_lr8",
            "204M": "screen_e_chorus_203m_lr4", "349M": "screen_e_chorus_349m_lr4"}
TR_TAGS = ("tr0p33", "tr0p11", "tr0p04")
TR_NAME = re.compile("^(?:" + "|".join(TR_STEMS.values()) + ")_(?:" + "|".join(TR_TAGS) + r")_s\d+$")
# Plain and Chorus at 47M and the base rate with three vocabularies.
VOCAB_RUNS = {"8k": ("screen_b_s1337", "screen_e_chorus_mb8_s1337"),
              "32k": ("screen_b_32k_s1337", "screen_e_chorus_32k_s1337"),
              "48k": ("screen_b_48k_s1337", "screen_e_chorus_48k_s1337")}


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def quiz_rows(run: str, zero: bool = False):
    """Item -> whether the model ranks the right answer first, or None without a quiz."""
    suffix = "_zero_all" if zero else ""
    for p in (QUIZ / f"{run}{suffix}.json", ROOT / "metrics" / "fact_quiz_screen" / f"{run}{suffix}.json",
              ROOT / "metrics" / "fact_quiz_screen" / f"rented5090_{run}{suffix}.json"):
        r = load(p)
        if r and r.get("rows"):
            # Questions without a candidate list (rank null: the numbers-in-words group) cannot be
            # ranked and are left out, so the score is over the ranked questions only.
            return {(row["group"], row["index"], row.get("frame")): row.get("rank") == 1 for row in r["rows"]
                    if row.get("rank") is not None}
    return None


def quiz(run: str, zero: bool = False):
    rows = quiz_rows(run, zero)
    return sum(rows.values()) / len(rows) if rows else None


def sign_test(a: dict, b: dict):
    """Items only `a` gets right, items only `b` gets right, and the two-sided exact sign-test p."""
    keys = a.keys() & b.keys()
    na = sum(a[k] and not b[k] for k in keys)
    nb = sum(b[k] and not a[k] for k in keys)
    n = na + nb
    if n == 0:
        return na, nb, 1.0
    tail = sum(math.comb(n, i) for i in range(min(na, nb) + 1)) / 2 ** n
    return na, nb, min(1.0, 2 * tail)


def throughput(d: Path):
    """Median tokens per second over the run, the first tenth (warmup, compilation) left out."""
    f = next(iter(d.glob("*_train.csv")), None)
    if not f:
        return None
    v = [float(r["tok_s"]) for r in list(csv.DictReader(open(f)))[1:] if r.get("tok_s")]
    v = v[len(v) // 10:]
    return statistics.median(v) if v else None


def runs() -> list[dict]:
    out = []
    for d in sorted(SCREEN.glob("screen_*")):
        m = NAME.match(d.name)
        ev = load(d / "eval_final.json")
        code = (d / "exit_code").read_text().strip() if (d / "exit_code").exists() else ""
        if not m or not ev or code != "0":
            continue
        bpb = load(d / "eval_bpb.json")
        man = load(next(iter(d.glob("*_manifest.json")), Path("/nonexistent"))) or {}
        out.append({
            "run": d.name, "arm": m["arm"], "table": TABLES[m["tab"] or ""] if m["arm"] == "e_chorus" else "",
            "size": SIZES[m["size"] or ""], "lr": float((m["lr"] or "1").replace("p", ".")),
            "seed": int(m["seed"]), "loss": ev["by_subset"]["full"]["cross_entropy_nats"],
            "bpb": bpb["by_subset"]["full"]["bits_per_byte"] if bpb else None,
            "params": (man.get("params_outside_table") or 0) + (man.get("params_table") or 0),
            "params_table": man.get("params_table") or 0,
            "mb": int(re.search(r"_mb(\d+)_s\d+$", d.name).group(1)) if re.search(r"_mb(\d+)_s\d+$", d.name) else None,
            "quiz": quiz(d.name), "quiz_zero": quiz(d.name, True) if m["arm"] == "e_chorus" else None,
            "tok_s": throughput(d), "card": (man.get("env") or {}).get("device_name"),
        })
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def cells(rs: list[dict]) -> dict:
    """(size, arm, table, lr) -> averaged record with the number of seeds."""
    g = defaultdict(list)
    for r in rs:
        g[(r["size"], r["arm"], r["table"], r["lr"])].append(r)
    out = {}
    for k, v in g.items():
        out[k] = {"loss": mean([r["loss"] for r in v]), "bpb": mean([r["bpb"] for r in v]),
                  "quiz": mean([r["quiz"] for r in v]), "quiz_zero": mean([r["quiz_zero"] for r in v]),
                  "params": v[0]["params"], "params_table": v[0].get("params_table", 0), "seeds": len(v)}
    return out


def equivalent(loss: float, line: list[tuple[float, float]]):
    """Size at which the b line reaches `loss`; (size, extrapolated)."""
    pts = sorted(line)
    if len(pts) < 2:
        return None, True
    for (n0, l0), (n1, l1) in zip(pts, pts[1:]):
        if l1 <= loss <= l0:
            f = (l0 - loss) / (l0 - l1)
            return math.exp(math.log(n0) + f * (math.log(n1) - math.log(n0))), False
    (n0, l0), (n1, l1) = (pts[-2], pts[-1]) if loss < pts[-1][1] else (pts[0], pts[1])
    f = (l0 - loss) / (l0 - l1)
    return math.exp(math.log(n0) + f * (math.log(n1) - math.log(n0))), True


def fmt(x, nd=4):
    return "$\\cdot$" if x is None else f"{x:.{nd}f}"


LRS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def best_rate(c: dict, size: str, arm: str, tab: str):
    """Measured rate with the lowest loss for (size, arm, tab); (rate, bracketed), bracketed when a
    lower and a higher measured rate are both worse. Rates within TIE of the lowest loss count as level,
    and among level rates the one averaged over more seeds is taken (then the lower loss)."""
    rates = sorted(lr for (s, a, t, lr) in c if (s, a, t) == (size, arm, tab))
    if not rates:
        return None, False
    lo = min(c[(size, arm, tab, x)]["loss"] for x in rates)
    level = [x for x in rates if c[(size, arm, tab, x)]["loss"] - lo < TIE]
    best = max(level, key=lambda x: (c[(size, arm, tab, x)]["seeds"], -c[(size, arm, tab, x)]["loss"]))
    return best, rates[0] < best < rates[-1]


def views(c: dict) -> dict:
    """Cells per (size, arm, table) under two settings. "x1": the ladder's rate for every arm. "best":
    b at its lowest measured rate; b_ve and every Chorus table at the lowest measured rate of Chorus
    x1.09 of the same size (only that table is swept), an arm missing at that rate left out."""
    out = {"x1": {}, "best": {}}
    for (size, arm, tab, lr), r in c.items():
        if lr == 1.0:
            out["x1"][(size, arm, tab)] = {**r, "lr": lr, "bracketed": None}
    for size in ORDER:
        rb = best_rate(c, size, "b", "")
        rc = best_rate(c, size, "e_chorus", "1.09")
        for (s, arm, tab, lr), r in c.items():
            want, ok = rb if arm == "b" else rc
            if s == size and want is not None and lr == want:
                out["best"][(size, arm, tab)] = {**r, "lr": lr, "bracketed": ok}
        # The fixed part only and the other table sizes were run at Chorus's best rate as it stood when
        # they started; where Chorus's best rate has since moved, they keep their own searched rate.
        if rc[0] is not None:
            for (s, arm, tab, lr), r in c.items():
                if s != size or arm == "b" or (arm == "e_chorus" and tab == "1.09") or lr == 1.0:
                    continue
                if (size, arm, tab) not in out["best"]:
                    out["best"][(size, arm, tab)] = {**r, "lr": lr, "bracketed": None, "inherited": True}
    return out


def b_line(v: dict) -> list[tuple[float, float]]:
    return [(v[(s, "b", "")]["params"], v[(s, "b", "")]["loss"]) for s in ORDER if (s, "b", "") in v]


def rate_label(r: dict) -> str:
    return muon(r["lr"]) + ("$^\\ddagger$" if r.get("bracketed") is False else "")


BASE_MUON = 0.015
TIE = 0.001  # losses closer than a thousandth of a nat count as level


def muon(factor: float) -> str:
    """The Muon learning rate at a multiple of the base rate (Muon 0.015, AdamW 0.0003, final 0.00003)."""
    return f"{BASE_MUON * factor:g}"


def main() -> None:
    rs = runs()
    c = cells(rs)
    v = views(c)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)

    # Learning rate per size: b and Chorus x1.09.
    lrs = LRS
    lines = ["\\begin{tabular}{ll" + "r" * len(lrs) + "}", "\\toprule",
             f" & & \\multicolumn{{{len(lrs)}}}{{c}}{{Peak Muon learning rate (AdamW in proportion)}} \\\\",
             f"\\cmidrule(l){{3-{len(lrs) + 2}}}",
             "Size & Model & " + " & ".join(muon(x) for x in lrs) + " \\\\", "\\midrule"]
    for size in ORDER:
        rows = []
        for arm, tab, label in (("b", "", "plain"), ("e_chorus", "1.09", "Chorus 1.09")):
            vals = [c.get((size, arm, tab, x), {}).get("loss") for x in lrs]
            if not any(v is not None for v in vals):
                continue
            pick, _ = best_rate(c, size, arm, tab)
            rows.append(f"{size} & {label} & " + " & ".join(
                "$\\cdot$" if v is None else (f"\\textbf{{{v:.4f}}}" if x == pick else f"{v:.4f}")
                for x, v in zip(lrs, vals)) + " \\\\")
        gains = [(c[(size, "b", "", x)]["loss"] - c[(size, "e_chorus", "1.09", x)]["loss"])
                 if (size, "b", "", x) in c and (size, "e_chorus", "1.09", x) in c else None for x in lrs]
        if any(g is not None for g in gains):
            rows.append(f"{size} & \\emph{{gain}} & " + " & ".join("" if g is None else f"\\emph{{{g:.3f}}}" for g in gains) + " \\\\")
        if rows:
            rows.append("\\addlinespace")
        lines += rows
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tables" / "lr.tex").write_text("\n".join(lines) + "\n")

    by_cell = defaultdict(list)
    for r in rs:
        by_cell[(r["size"], r["arm"], r["table"], r["lr"])].append(round(r["loss"], 4))
    spread_max = max((max(g) - min(g) for g in by_cell.values() if len(g) > 1), default=0.0)


    # What a table is worth: first at each arm's lowest measured rate (the headline), then at the
    # ladder's rate for every arm (sensitivity).
    worth = ["\\begin{tabular}{lllrrrrrrr}", "\\toprule",
             "Size & Model & Muon rate & Seeds & Loss & Gain & Equivalent size & Multiplier & Quiz & Zeroed quiz \\\\"]
    for name, title in (("best", "Each model at its best measured rate"), ("x1", "Every model at the base rate")):
        view = v[name]
        line = b_line(view)
        worth += ["\\midrule", f"\\multicolumn{{10}}{{l}}{{\\emph{{{title}}}}} \\\\", "\\midrule"]
        for size in ORDER:
            b = view.get((size, "b", ""))
            if not b:
                continue
            worth.append(f"{size} & plain & {rate_label(b)} & {b['seeds']} & {fmt(b['loss'])} & & & & {fmt(b['quiz'], 3)} & \\\\")
            arms = [("b_ve", "", "fixed part only")] + [("e_chorus", t, None) for t in ("0.275", "1.09", "4.35", "max")]
            for arm, tab, label in arms:
                r = view.get((size, arm, tab))
                if not r:
                    continue
                if label is None:
                    label = f"Chorus {(r['params'] - b['params']) / b['params']:.3g}"
                eq, ext = equivalent(r["loss"], line)
                mult = eq / b["params"] if eq else None
                mark = "$^\\dagger$" if ext else ""
                head = f"{size} & {label} & {rate_label(r)} & {r['seeds']} & {fmt(r['loss'])} & {b['loss'] - r['loss']:.4f} & "
                worth.append(head + (f"{eq / 1e6:.0f}M{mark} & $\\times${mult:.2f}{mark} & " if eq else " & & ")
                             + f"{fmt(r['quiz'], 3)} & {fmt(r['quiz_zero'], 3)} \\\\")
            worth.append("\\addlinespace")
    worth += ["\\bottomrule", "\\end{tabular}"]

    # Points for the figure, at each arm's lowest measured rate: multiplier against the table's added
    # parameters relative to the backbone, one file per backbone size (Chorus arms only), read by
    # pgfplots in main.tex.
    best = v["best"]
    for size in ORDER:
        b = best.get((size, "b", ""))
        pts = []
        line = b_line(best)
        for tab in ("0.275", "1.09", "4.35", "max"):
            r = best.get((size, "e_chorus", tab))
            if not (b and r):
                continue
            eq, ext = equivalent(r["loss"], line)
            if eq:
                pts.append(f"{(r['params'] - b['params']) / b['params']:.4f} {eq / b['params']:.4f} {int(ext)}")
        (OUT / "tables" / f"points_{size}.dat").write_text("ratio mult extrapolated\n" + "\n".join(pts) + "\n")
    # The same points as pgfplots commands, only for sizes that have any (black and white: one line
    # style and marker per backbone size; extrapolated points as open markers, not joined).
    # Colour per backbone size (colour names defined in main.tex), with line style and marker kept as
    # a second encoding, so the figure still reads in black and white and for colour-blind readers.
    styles = {"16M": ("dotted", "oplus*"), "47M": ("solid", "square*"), "107M": ("dashed", "triangle*"),
              "204M": ("dashdotted", "diamond*"), "349M": ("densely dashed", "pentagon*"), "550M": ("loosely dotted", "x"), "818M": ("loosely dashed", "star")}
    colour = {s: "c" + s.lower() for s in styles}
    # Points of different sizes at the same ratio would cover each other, so each size is shifted
    # sideways by a fixed fraction of a doubling (stated in the caption; the true ratios are in the
    # points files and the worth table).
    dodge = {s: 2 ** (0.15 * (k - 2)) for k, s in enumerate(ORDER)}
    plots = []
    for size in ORDER:
        rows = [l.split() for l in (OUT / "tables" / f"points_{size}.dat").read_text().splitlines()[1:] if l.strip()]
        rows = [(f"{float(x) * dodge[size]:.4f}", y, e) for x, y, e in rows]
        meas = [(x, y) for x, y, e in rows if e == "0"]
        extra = [(x, y) for x, y, e in rows if e == "1"]
        dash, mark = styles[size]
        if meas:
            plots.append(f"\\addplot[{colour[size]}, thick, {dash}, mark={mark}, mark options={{solid}}] coordinates {{" + " ".join(f"({x},{y})" for x, y in meas) + "};")
            plots.append(f"\\addlegendentry{{{size}}}")
        if extra:
            open_mark = mark.rstrip("*")
            plots.append(f"\\addplot[{colour[size]}, thick, only marks, mark={open_mark}, mark options={{fill=white, solid}}] coordinates {{" +
                         " ".join(f"({x},{y})" for x, y in extra) + "};")
            plots.append(f"\\addlegendentry{{{size} (extrapolated)}}")
    (OUT / "tables" / "worth_plots.tex").write_text("\n".join(plots) + "\n")

    # Teaser, left panel, at each arm's lowest measured rate: held-out loss against backbone size for the
    # models without a table, and the x1.09 Chorus models with an arrow to the size at which the b line
    # reaches their loss.
    line = b_line(best)
    bl = [(n / 1e6, l) for n, l in line]
    teaser = []
    if bl:
        teaser.append("\\addplot[black!75, thick, solid, mark=*] coordinates {" + " ".join(f"({n:.1f},{l:.4f})" for n, l in bl) + "};")
        teaser.append("\\addlegendentry{plain model}")
    ch = []
    for s in ORDER:
        b, r = best.get((s, "b", "")), best.get((s, "e_chorus", "1.09"))
        if b and r:
            eq, ext = equivalent(r["loss"], line)
            ch.append((b["params"] / 1e6, r["loss"], eq / 1e6 if eq else None, ext))
    sizes_ch = [s for s in ORDER if best.get((s, "b", "")) and best.get((s, "e_chorus", "1.09"))]
    for k, (s, (n, l, eq, ext)) in enumerate(zip(sizes_ch, ch)):
        teaser.append(f"\\addplot[c{s.lower()}, only marks, mark=square*, mark size=2.5pt, forget plot] "
                      f"coordinates {{({n:.1f},{l:.4f})}};")
        if eq:
            style = "densely dotted" if ext else "solid"
            teaser.append(f"\\draw[->, thick, c{s.lower()}, {style}] (axis cs:{n:.1f},{l:.4f}) -- (axis cs:{eq:.1f},{l:.4f});")
    if ch:
        # Legend in a neutral grey: the colour of a Chorus point is its backbone size, as in panel (b).
        teaser.append("\\addlegendimage{only marks, mark=square*, mark size=2.5pt, black!45}")
        teaser.append("\\addlegendentry{Chorus, equal-size tables}")
        teaser.append("\\addlegendimage{black!45, thick, ->}")
        teaser.append("\\addlegendentry{to the size of equal loss}")
        if any(ext for _, _, _, ext in ch):
            teaser.append("\\addlegendimage{black!45, thick, densely dotted, ->}")
            teaser.append("\\addlegendentry{beyond the largest plain model}")
    (OUT / "tables" / "teaser_loss.tex").write_text("\n".join(teaser) + "\n")
    (OUT / "tables" / "worth.tex").write_text("\n".join(worth) + "\n")

    # Every run.
    # A longtable, so the list can run over pages; its header repeats on each page.
    head = "Size & Model & Muon rate & Seed & Parameters & Loss & Bits/byte & Quiz & Zeroed & Tokens/s & Micro & Card \\\\"
    allr = ["\\begin{longtable}{lllrrrrrrrrl}",
            "\\caption{Every run of the rate search and of the table sizes. Muon rate: the backbone's Muon "
            "learning rate at its peak (AdamW in proportion). Chorus $r$: added ratio $r$. Parameters: all of the model's, Chorus's tables included. Loss: held-out nats per "
            "token. Quiz: share of the 688 scored questions known; zeroed: with the N-gram tables set to zero. Tokens/s: "
            "training throughput (Section~\\ref{sec:measure}). Micro: micro-batch in sequences, the rest of the 256 per "
            "step made up by accumulation. $\\cdot$: not measured.}"
            "\\label{tab:runs} \\\\", "\\toprule", head, "\\midrule", "\\endfirsthead",
            "\\toprule", head, "\\midrule", "\\endhead", "\\bottomrule", "\\endlastfoot"]
    for r in sorted(rs, key=lambda r: (ORDER.index(r["size"]), r["arm"], r["table"], r["lr"], r["seed"])):
        bp = next((v["params"] for (s, a, t, _), v in c.items() if s == r["size"] and a == "b"), None)
        ratio = f"{(r['params'] - bp) / bp:.3g}" if bp and r["arm"] == "e_chorus" else r["table"]
        label = {"b": "plain", "b_ve": "fixed part only"}.get(r["arm"], f"Chorus {ratio}")
        card = (r["card"] or "").replace("NVIDIA GeForce ", "")
        tok = f"{r['tok_s'] / 1e3:.0f}k" if r["tok_s"] else ""
        allr.append(f"{r['size']} & {label} & {muon(r['lr'])} & {r['seed']} & {r['params'] / 1e6:.1f}M & "
                    f"{fmt(r['loss'])} & {fmt(r['bpb'])} & {fmt(r['quiz'], 3)} & {fmt(r['quiz_zero'], 3)} & "
                    f"{tok} & {r['mb'] or (16 if r['size'] in ('16M', '47M', '107M') else 8)} & {card} \\\\")
    allr += ["\\end{longtable}"]
    (OUT / "tables" / "runs.tex").write_text("\n".join(allr) + "\n")

    # Cost per token: tokens per second of each Chorus arm against b of the same size on the same card
    # type, one row per size and table (median over its runs; RTX 5090 preferred).
    cost_rows = []
    vals = {"runs": str(len(rs))}
    rel_all = []
    for size in ORDER:
        bs = [r for r in rs if r["size"] == size and r["arm"] == "b" and r["tok_s"] and r["card"]]
        b0 = next((c[(size, "b", "", 1.0)] for _ in [0] if (size, "b", "", 1.0) in c), None)
        for tab in ("0.275", "1.09", "4.35", "max"):
            group = [r for r in rs if r["size"] == size and r["arm"] == "e_chorus" and r["table"] == tab and r["tok_s"] and r["card"]]
            cards = sorted({r["card"] for r in group}, key=lambda x: ("5090" not in x, x))
            for card in cards:
                ref = [b["tok_s"] for b in bs if b["card"] == card]
                mine = [r["tok_s"] for r in group if r["card"] == card]
                if not ref or not mine:
                    continue
                rel = statistics.median(mine) / statistics.median(ref)
                rel_all.append(rel)
                ratio = (group[0]["params"] - b0["params"]) / b0["params"] if b0 else None
                label = f"Chorus {ratio:.3g}" if ratio else f"Chorus {tab}"
                cost_rows.append((size, tab, f"{size} & {label} & {card.replace('NVIDIA GeForce ', '')} & "
                                  f"{statistics.median(mine) / 1e3:.0f}k & {statistics.median(ref) / 1e3:.0f}k & {rel:.2f}",
                                  group[0]))
                break
    if rel_all:
        lo, hi = f"{(1 - max(rel_all)) * 100:.0f}", f"{(1 - min(rel_all)) * 100:.0f}"
        vals["slowdown"] = lo if lo == hi else f"{lo} to {hi}"

    # Table precision after training.
    qrows = ["\\begin{tabular}{llrrr}", "\\toprule", "Table & Bits & Loss change & Quiz & Table size \\\\", "\\midrule"]
    deltas = defaultdict(list)
    qdeltas = defaultdict(list)
    kept = defaultdict(list)
    qorder = ["screen_e_chorus_mb8_s1337", "screen_e_chorus_big_mb8_s1337", "screen_e_chorus_huge_s2"]
    found = {p.name.rsplit("_", 1)[0] for p in (QUANT / "eval").glob("*_intact.json")} if (QUANT / "eval").exists() else set()
    for base in [b for b in qorder if b in found] + sorted(found - set(qorder)):
        intact = load(QUANT / "eval" / f"{base}_intact.json")
        if not intact:
            continue
        l0 = intact["by_subset"]["full"]["cross_entropy_nats"]
        label = {"screen_e_chorus_mb8_s1337": "47M, Chorus 1.09", "screen_e_chorus_big_mb8_s1337": "47M, Chorus 4.35",
                 "screen_e_chorus_huge_s2": "47M, Chorus 21.7"}.get(base, base.replace("_", "\\_"))
        for v in ("intact", "bits8", "bits4", "bits3", "bits2"):
            e = load(QUANT / "eval" / f"{base}_{v}.json")
            if not e:
                continue
            l = e["by_subset"]["full"]["cross_entropy_nats"]
            qz = load(QUANT / "quiz" / f"{base}_{v}.json")
            ranked = [r for r in qz["rows"] if r.get("rank") is not None] if qz and qz.get("rows") else []
            qv = sum(r["rank"] == 1 for r in ranked) / len(ranked) if ranked else None
            qt = e.get("quantized_table") or {}
            size = f"{qt['bytes'] / 1e6:.0f} MB (bf16 {qt['bytes_bf16'] / 1e6:.0f} MB)" if qt else ""
            bits = "32 (trained)" if v == "intact" else v[4:]
            qrows.append(f"{label} & {bits} & {'' if v == 'intact' else f'{l - l0:+.4f}'} & {fmt(qv, 3)} & {size} \\\\")
            if v != "intact":
                deltas[v].append(l - l0)
                q0 = load(QUANT / "quiz" / f"{base}_intact.json")
                r0 = [r for r in q0["rows"] if r.get("rank") is not None] if q0 and q0.get("rows") else []
                if r0 and ranked:
                    qdeltas[v].append(abs(sum(r["rank"] == 1 for r in ranked) - sum(r["rank"] == 1 for r in r0)))
                tab = {"screen_e_chorus_mb8_s1337": "1.09", "screen_e_chorus_big_mb8_s1337": "4.35",
                       "screen_e_chorus_huge_s2": "max"}.get(base)
                ch, b1 = c.get(("47M", "e_chorus", tab, 1.0)), c.get(("47M", "b", "", 1.0))
                if ch and b1:
                    kept[v].append(1 - (l - l0) / (b1["loss"] - ch["loss"]))
        qrows.append("\\addlinespace")
    qrows += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tables" / "quant.tex").write_text("\n".join(qrows) + "\n")
    for v, d in deltas.items():
        lo, hi = f"{min(d):.3f}", f"{max(d):.3f}"
        vals[f"quant-{v}"] = lo if lo == hi else f"{lo} to {hi}"
    for v, d in qdeltas.items():
        vals[f"quantquiz-{v}"] = str(max(d))
    for v, k in kept.items():
        lo, hi = f"{min(k) * 100:.0f}", f"{max(k) * 100:.0f}"
        vals[f"quantkept-{v}"] = lo if lo == hi else f"{lo} to {hi}"
    # Seed-to-seed spread: over every configuration trained with more than one seed, the range of its
    # held-out loss and of its quiz score across seeds.
    groups = defaultdict(list)
    for r in rs:
        groups[(r["size"], r["arm"], r["table"], r["lr"])].append(r)
    multi = {k: g for k, g in groups.items() if len(g) > 1}
    if multi:
        vals["seedcells"] = str(len(multi))
        ss = [s for s in ORDER if any(k[0] == s for k in multi)]
        vals["seedsizes"] = ss[0] if len(ss) == 1 else ", ".join(ss[:-1]) + " and " + ss[-1]
        spread = [max(round(r["loss"], 4) for r in g) - min(round(r["loss"], 4) for r in g) for g in multi.values()]
        vals["seedspread-loss"] = f"{min(spread):.4f} to {max(spread):.4f}"
        qs = [[r["quiz"] for r in g if r["quiz"] is not None] for g in multi.values()]
        qs = [max(q) - min(q) for q in qs if len(q) > 1]
        if qs:
            vals["seedspread-quiz"] = f"{min(qs):.3f} to {max(qs):.3f}"
    for (size, arm, tab, lr), r in c.items():
        b = c.get((size, "b", "", lr))
        key = f"{size}-{'b' if arm == 'b' else 'bve' if arm == 'b_ve' else 'chorus' + tab}-lr{lr:g}"
        vals[f"loss-{key}"] = f"{r['loss']:.4f}"
        if r["quiz"] is not None:
            vals[f"quiz-{key}"] = f"{r['quiz']:.3f}"
        if r["quiz_zero"] is not None:
            vals[f"quizzero-{key}"] = f"{r['quiz_zero']:.3f}"
        if b and arm != "b":
            vals[f"gain-{key}"] = f"{b['loss'] - r['loss']:.3f}"
            line = [(c[(s, "b", "", lr)]["params"], c[(s, "b", "", lr)]["loss"]) for s in ORDER if (s, "b", "", lr) in c]
            eq, ext = equivalent(r["loss"], line)
            if eq:
                vals[f"mult-{key}"] = f"{eq / b['params']:.2f}" + ("$^\\dagger$" if ext else "")
    # The same values at each arm's lowest measured rate, keys ending in "-best", and the rates chosen.
    line = b_line(best)
    for (size, arm, tab), r in best.items():
        b = best.get((size, "b", ""))
        key = f"{size}-{'b' if arm == 'b' else 'bve' if arm == 'b_ve' else 'chorus' + tab}-best"
        vals[f"loss-{key}"] = f"{r['loss']:.4f}"
        vals[f"rate-{key}"] = rate_label(r)
        if r["quiz"] is not None:
            vals[f"quiz-{key}"] = f"{r['quiz']:.3f}"
        if r["quiz_zero"] is not None:
            vals[f"quizzero-{key}"] = f"{r['quiz_zero']:.3f}"
        if b and arm != "b":
            vals[f"gain-{key}"] = f"{b['loss'] - r['loss']:.3f}"
            eq, ext = equivalent(r["loss"], line)
            if eq:
                vals[f"mult-{key}"] = f"{eq / b['params']:.2f}" + ("$^\\dagger$" if ext else "")
    # What the N-gram rows add over the fixed part alone: Chorus x1.09 against b_ve, per size, at the
    # ladder's rate ("-lr1") and at the tuned rates ("-best").
    for size in ORDER:
        for tag, ch, ve in (("lr1", c.get((size, "e_chorus", "1.09", 1.0)), c.get((size, "b_ve", "", 1.0))),
                            ("best", best.get((size, "e_chorus", "1.09")), best.get((size, "b_ve", "")))):
            if ch and ve:
                vals[f"rows-{size}-{tag}"] = f"{ve['loss'] - ch['loss']:.3f}"
    # Quiz at the tuned rates, seed 1337: Chorus x1.09 against b of the same size, and Chorus against
    # itself with its N-gram tables zeroed; items only the first gets right, only the second, sign-test p.
    by = {(r["size"], r["arm"], r["table"], r["lr"], r["seed"]): r["run"] for r in rs}
    pooled, small = [0, 0], [0, 0]
    for size in ORDER:
        rb, rc = best.get((size, "b", "")), best.get((size, "e_chorus", "1.09"))
        if not rb or not rc:
            continue
        qc = quiz_rows(by.get((size, "e_chorus", "1.09", rc["lr"], 1337), "none"))
        qb = quiz_rows(by.get((size, "b", "", rb["lr"], 1337), "none"))
        qz = quiz_rows(by.get((size, "e_chorus", "1.09", rc["lr"], 1337), "none"), True)
        for tag, other in (("sign", qb), ("signzero", qz)):
            if qc and other:
                na, nb, p = sign_test(qc, other)
                pf = "<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
                vals[f"{tag}-{size}"] = f"{na} against {nb}, $p{'' if pf.startswith('<') else '='}{pf}$"
                vals[f"{tag}a-{size}"], vals[f"{tag}b-{size}"] = str(na), str(nb)
                vals[f"{tag}p-{size}"] = f"${pf}$"
                if tag == "sign":
                    pooled[0] += na
                    pooled[1] += nb
                    if size in ("16M", "47M", "107M"):
                        small[0] += na
                        small[1] += nb
    # Where Chorus's loss is level (within TIE) over several rates, the quiz test against the plain model
    # of its size at each of those rates, seed 1337.
    for size in ORDER:
        rb = best.get((size, "b", ""))
        rates = sorted(lr for (s, a, tb, lr) in c if s == size and a == "e_chorus" and tb == "1.09" and lr != 1.0)
        if not rb or not rates:
            continue
        lo = min(c[(size, "e_chorus", "1.09", x)]["loss"] for x in rates)
        level = [x for x in rates if c[(size, "e_chorus", "1.09", x)]["loss"] - lo < TIE]
        qb = quiz_rows(by.get((size, "b", "", rb["lr"], 1337), "none"))
        parts = []
        for x in level:
            qc = quiz_rows(by.get((size, "e_chorus", "1.09", x, 1337), "none"))
            if qc and qb:
                na, nb, pp = sign_test(qc, qb)
                pf = "<0.001" if pp < 0.001 else f"{pp:.3f}" if pp < 0.01 else f"{pp:.2f}"
                parts.append(f"{na} against {nb} at {muon(x)} ($p{'' if pf.startswith('<') else '='}{pf}$)")
        if len(parts) > 1:
            vals[f"signlevel-{size}"] = ", ".join(parts[:-1]) + " and " + parts[-1]
    if sum(pooled):
        n, k = sum(pooled), min(pooled)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
        vals["sign-pooled"] = f"{pooled[0]} against {pooled[1]}, " + ("$p<0.001$" if p < 0.001 else f"$p={p:.3f}$")
    if sum(small):
        n, k = sum(small), min(small)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
        vals["sign-pooled-small"] = f"{small[0]} against {small[1]}, " + ("$p<0.001$" if p < 0.001 else f"$p={p:.3f}$")
    # The quiz's own noise: two seeds of the same configuration (b at its best rate), and Chorus against
    # b's second seed.
    for size in ORDER:
        rb, rc = best.get((size, "b", "")), best.get((size, "e_chorus", "1.09"))
        if not rb:
            continue
        q1 = quiz_rows(by.get((size, "b", "", rb["lr"], 1337), "none"))
        q2 = quiz_rows(by.get((size, "b", "", rb["lr"], 2), "none"))
        if q1 and q2:
            na, nb, p = sign_test(q1, q2)
            vals[f"signseed-{size}"] = f"{na} against {nb}, " + ("$p<0.001$" if p < 0.001 else f"$p={p:.3f}$" if p < 0.01
                                                                 else f"$p={p:.2f}$")
        qc = quiz_rows(by.get((size, "e_chorus", "1.09", rc["lr"], 1337), "none")) if rc else None
        if qc and q2:
            na, nb, p = sign_test(qc, q2)
            vals[f"sign2-{size}"] = f"{na} against {nb}, " + ("$p<0.001$" if p < 0.001 else f"$p={p:.2f}$")
    # Side runs at 47M and the ladder's rate: the tables' own rate (lr_mult, 5 by default, so the
    # factor is relative to the default) and the authors' Ember optimiser for the tables.
    def final_loss(run: str):
        ev = load(SCREEN / run / "eval_final.json")
        return ev["by_subset"]["full"]["cross_entropy_nats"] if ev else None
    for k, names in SIDE_RUNS.items():
        ls = [x for x in (final_loss(n) for n in names) if x is not None]
        if ls:
            vals[f"side-{k}"] = f"{mean(ls):.4f}"
    if ("47M", "e_chorus", "1.09", 1.0) in c:
        vals["side-tablerate1"] = f"{c[('47M', 'e_chorus', '1.09', 1.0)]['loss']:.4f}"
    # The 47M multiplier with a third of the tables' rate at Chorus's best backbone rate (seed 1337),
    # read on the same plain line as the best-rate multipliers.
    tr = final_loss("screen_e_chorus_lr8_tr0p33_s1337")
    if tr is not None and ("47M", "b", "") in best:
        eq, ext = equivalent(tr, b_line(best))
        if eq:
            vals["mult-47M-tr0p33"] = f"{eq / best[('47M', 'b', '')]['params']:.2f}" + ("$^\\dagger$" if ext else "")
    # A lower tables' rate (lr_mult divided by 3, 9 or 27) at Chorus's best backbone rate: the loss change
    # against the default tables' rate for the same seeds, and both multipliers from those seeds.
    third_diffs, third_mults = [], []
    for size, stem in TR_STEMS.items():
        if (size, "b", "") not in best:
            continue
        for tag in TR_TAGS:
            pairs = [(final_loss(f"{stem}_{tag}_s{s}"), final_loss(f"{stem}_s{s}")) for s in (1337, 2, 3)]
            pairs = [(t, d) for t, d in pairs if t is not None and d is not None]
            if not pairs:
                continue
            diff = mean(d - t for t, d in pairs)
            vals[f"trdiff-{size}-{tag}"] = f"{diff:.4f}"
            vals[f"trseeds-{size}-{tag}"] = str(len(pairs))
            mults = []
            for key, loss in ((f"trloss-{size}-{tag}", mean(t for t, _ in pairs)),
                              (f"trlossdef-{size}-{tag}", mean(d for _, d in pairs))):
                vals[key] = f"{loss:.4f}"
                eq, ext = equivalent(loss, b_line(best))
                if eq:
                    vals[key.replace("trloss", "treq")] = (f"{eq / 1e6:.1f}M" if eq < 1e8 else f"{eq / 1e6:.0f}M") + ("$^\\dagger$" if ext else "")
                mults.append((eq / best[(size, "b", "")]["params"], ext) if eq else None)
            if mults[0] and mults[1]:
                vals[f"trmult-{size}-{tag}"] = f"{mults[0][0]:.2f}" + ("$^\\dagger$" if mults[0][1] else "")
                vals[f"trmultdef-{size}-{tag}"] = f"{mults[1][0]:.2f}" + ("$^\\dagger$" if mults[1][1] else "")
                if tag == "tr0p33":
                    third_diffs.append(diff)
                    third_mults.append(mults[0][0] - mults[1][0])
    if third_diffs:
        vals["trdiff-range"] = f"{min(third_diffs):.3f} to {max(third_diffs):.3f}"
        vals["trmultup-range"] = f"{min(third_mults):.2f} to {max(third_mults):.2f}"
    # Chorus's gain in bits per byte at 47M and the base rate with three vocabularies, one seed each,
    # the same table parameters (a larger token table leaves fewer N-gram rows at 32k and 48k).
    for tag, (plain_run, chorus_run) in VOCAB_RUNS.items():
        pb, cb = load(SCREEN / plain_run / "eval_bpb.json"), load(SCREEN / chorus_run / "eval_bpb.json")
        if pb and cb:
            vals[f"vocabgain-{tag}"] = (f"{pb['by_subset']['full']['bits_per_byte'] - cb['by_subset']['full']['bits_per_byte']:.3f}")
    # The plain model alone with a larger vocabulary, against the 8,000-token plain model, in bits per byte.
    b8 = load(SCREEN / VOCAB_RUNS["8k"][0] / "eval_bpb.json")
    for tag in ("32k", "48k"):
        bv = load(SCREEN / VOCAB_RUNS[tag][0] / "eval_bpb.json")
        if b8 and bv:
            vals[f"vocabplain-{tag}"] = f"{b8['by_subset']['full']['bits_per_byte'] - bv['by_subset']['full']['bits_per_byte']:.3f}"
    # The fixed part's share of Chorus's gain with the plain model, the fixed part only and Chorus x1.09 all at
    # the backbone rate the fixed part only ran at (its own rate is not searched).
    shares = []
    for size in ORDER:
        fx = best.get((size, "b_ve", ""))
        if not fx or fx["lr"] == 1.0:
            continue
        pl, ch = c.get((size, "b", "", fx["lr"])), c.get((size, "e_chorus", "1.09", fx["lr"]))
        if pl and ch and pl["loss"] > ch["loss"]:
            shares.append(100 * (pl["loss"] - fx["loss"]) / (pl["loss"] - ch["loss"]))
    if shares:
        vals["fixedsame-range"] = f"{min(shares):.0f} to {max(shares):.0f}"
    # Other token budgets (scripts/run_branch_rented.sh): half the steps, from the full run's state at the
    # half run's last full-rate step, or from step 0 where the full run saved no such state (47M); and
    # twice the steps, from the full run's last full-rate step. Each arm at the rate chosen at the full
    # budget; both arms of a size use the same kind of half run. At each size every budget averages the
    # same seeds, those run at all of its budgets, and each budget is read on its own plain line. Tokens
    # per parameter count the plain model's parameters.
    braw = defaultdict(dict)
    for d in sorted(SCREEN.glob("screen_*")):
        m = BUDGET_NAME.match(d.name)
        ev = load(d / "eval_final.json")
        code = (d / "exit_code").read_text().strip() if (d / "exit_code").exists() else ""
        if not (m and ev and code == "0"):
            continue
        size, arm, lr = SIZES[m["size"] or ""], m["arm"], float(m["lr"])
        k = (size, arm, "1.09" if arm == "e_chorus" else "")
        if k not in best or best[k]["lr"] != lr:
            print(f"{d.name}: not at the full budget's best rate, left out")
            continue
        braw[(m["kind"], size, arm)][int(m["seed"])] = ev["by_subset"]["full"]["cross_entropy_nats"]
    per = defaultdict(dict)
    for r in rs:
        k = (r["size"], r["arm"], "1.09" if r["arm"] == "e_chorus" else "")
        if r["arm"] in ("b", "e_chorus") and r["table"] == k[2] and k in best and r["lr"] == best[k]["lr"]:
            per[("full", r["size"], r["arm"])][r["seed"]] = r["loss"]
    for size in ORDER:
        both = [kd for kd in ("half", "halffresh") if (kd, size, "b") in braw and (kd, size, "e_chorus") in braw]
        kind = both[0] if both else next((kd for kd in ("half", "halffresh") if (kd, size, "b") in braw), None)
        for arm in ("b", "e_chorus"):
            if kind and (kind, size, arm) in braw:
                per[("half", size, arm)] = braw[(kind, size, arm)]
            if ("double", size, arm) in braw:
                per[("double", size, arm)] = braw[("double", size, arm)]
    common = {}
    for size in ORDER:
        sets = [set(v) for (bg, s, a), v in per.items() if s == size]
        if sets:
            common[size] = set.intersection(*sets)

    def bval(bg, size, arm):
        v = per.get((bg, size, arm))
        xs = [v[s] for s in sorted(common.get(size, ())) if v and s in v]
        return mean(xs) if xs else None

    def tokens(run):
        man = load(next(iter((SCREEN / run).glob("*_manifest.json")), Path("/nonexistent"))) or {}
        cfg = load(ROOT / man.get("config", "/nonexistent")) or {}
        if not cfg:
            return None
        return cfg["batch_size"] * cfg["grad_accum"] * cfg["block_size"] * cfg["max_steps"]
    factor = {"half": 0.5, "full": 1.0, "double": 2.0}
    brows, bmults = [], {}
    for size in ORDER:
        b = best.get((size, "b", ""))
        if not b or not any((bg, size, "e_chorus") in per for bg in factor):
            continue
        tk = tokens(by.get((size, "b", "", b["lr"], 1337), ""))
        cols = {"tpp": [], "gain": [], "mult": []}
        for bg in factor:
            line_bg = [(best[(s, "b", "")]["params"], bval(bg, s, "b")) for s in ORDER
                       if (s, "b", "") in best and bval(bg, s, "b") is not None]
            lb, lc = bval(bg, size, "b"), bval(bg, size, "e_chorus")
            tpp = factor[bg] * tk / b["params"] if tk else None
            cols["tpp"].append(None if tpp is None or lb is None else f"{tpp:.0f}" if tpp >= 10 else f"{tpp:.1f}")
            if tpp is not None:
                vals[f"tpp-{size}-{bg}"] = f"{tpp:.0f}" if tpp >= 10 else f"{tpp:.1f}"
            if lb is None or lc is None:
                cols["gain"].append(None)
                cols["mult"].append(None)
                continue
            eq, ext = equivalent(lc, line_bg)
            mark = "$^\\dagger$" if ext else ""
            vals[f"bgain-{size}-{bg}"] = f"{lb - lc:.3f}"
            cols["gain"].append(f"{lb - lc:.3f}")
            if eq:
                vals[f"bmult-{size}-{bg}"] = f"$\\times${eq / b['params']:.2f}{mark}"
                if not ext:
                    bmults[(size, bg)] = round(eq / b["params"], 2)
            cols["mult"].append(vals.get(f"bmult-{size}-{bg}"))
        cell = lambda x: "$\\cdot$" if x is None else x
        brows.append(f"{size} & " + " & ".join(cell(x) for k in ("tpp", "gain", "mult") for x in cols[k]) + " \\\\")
        if len(common.get(size, ())) > 1:
            vals[f"bseeds-{size}"] = str(len(common[size]))
    (OUT / "tables" / "budget.tex").write_text("\n".join(
        ["\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
         " & \\multicolumn{3}{c}{Tokens per parameter} & \\multicolumn{3}{c}{Gain (nats)} & \\multicolumn{3}{c}{Multiplier} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-10}",
         "Size & Half & Full & Double & Half & Full & Double & Half & Full & Double \\\\", "\\midrule"]
        + brows + ["\\bottomrule", "\\end{tabular}"]) + "\n")
    # Half against full budget at every size with both, as printed (a value read beyond the largest
    # plain model included), so the range matches the table.
    bm = lambda s, bg: float(vals[f"bmult-{s}-{bg}"].replace("$\\times$", "").split("$")[0])
    diffs = [round(bm(s, "half") - bm(s, "full"), 2) for s in ORDER
             if f"bmult-{s}-half" in vals and f"bmult-{s}-full" in vals]
    if diffs:
        vals["bdiff-range"] = f"{min(diffs):.2f} to {max(diffs):.2f}"
    bseeds = [s for s in ORDER if len(common.get(s, ())) > 1]
    vals["bseeds"] = ("seed 1337" if not bseeds else "seed 1337, and at " + ", ".join(bseeds)
                      + " also seed " + ", ".join(str(x) for x in sorted(set.union(*(common[s] for s in bseeds)) - {1337})))
    if ("half", "16M", "b") in braw and ("halffresh", "16M", "b") in braw:
        vals["halfbranch-16M"] = f"{braw[('half', '16M', 'b')][1337]:.4f}"
        vals["halffresh-16M"] = f"{braw[('halffresh', '16M', 'b')][1337]:.4f}"
    # The 550M plain model's loss at its best rate as predicted before either 550M run started
    # (written down before either 550M run started): from the curve fitted to 16M to 349M, and from the
    # straight line through 204M and 349M on a logarithmic size axis. Fixed values, kept as written.
    vals["pred-550M-curve"] = "2.8982"
    vals["pred-550M-line"] = "2.8941"
    vals["pred-550M-inputs"] = "3.5489, 3.2831, 3.1257, 3.0232 and 2.9535"
    # How the reading of the multiplier depends on the method (best view, tables of 1.09): (a) on a
    # size axis without the tied embedding (8,000 x width), (b) on a curve L = E + A N^-alpha fitted to
    # the plain models by least squares (alpha on a grid, E and A linear) instead of straight segments;
    # and the exchange rate, backbone parameters replaced per parameter Chorus adds.
    width = {"16M": 384, "47M": 576, "107M": 768, "204M": 960, "349M": 1152, "550M": 1344, "818M": 1536}
    plain = [(s, best[(s, "b", "")]) for s in ORDER if (s, "b", "") in best]
    ne_line = [(r["params"] - 8000 * width[s], r["loss"]) for s, r in plain]
    ns = [r["params"] for _, r in plain]
    ys = [r["loss"] for _, r in plain]
    fit = None
    for k in range(10, 500):
        alpha = k / 500
        us = [n ** -alpha for n in ns]
        mu, my = statistics.mean(us), statistics.mean(ys)
        suu = sum((u - mu) ** 2 for u in us)
        if suu == 0:
            continue
        a = sum((u - mu) * (y - my) for u, y in zip(us, ys)) / suu
        e = my - a * mu
        sse = sum((e + a * u - y) ** 2 for u, y in zip(us, ys))
        if a > 0 and (fit is None or sse < fit[0]):
            fit = (sse, alpha, e, a)
    if fit:
        vals["fit-alpha"] = f"{fit[1]:.3f}"
        vals["fit-E"] = f"{fit[2]:.4f}"
        vals["fit-A"] = f"{fit[3] * 1e6 ** -fit[1]:.4f}"
    for size in ORDER:
        for tab in ("1.09", "max", "4.35", "0.275"):
            r, b = best.get((size, "e_chorus", tab)), best.get((size, "b", ""))
            if not (r and b) or size not in width:
                continue
            key = f"{size}-chorus{tab}"
            eq, ext = equivalent(r["loss"], line)
            if eq:
                vals[f"exch-{key}"] = (f"{(eq - b['params']) / (r['params'] - b['params']):.2f}"
                                       + ("$^\\dagger$" if ext else ""))
            nb = b["params"] - 8000 * width[size]
            eqn, extn = equivalent(r["loss"], ne_line)
            if eqn:
                vals[f"multne-{key}"] = f"{eqn / nb:.2f}" + ("$^\\dagger$" if extn else "")
            if fit and r["loss"] > fit[2]:
                eqf = ((r["loss"] - fit[2]) / fit[3]) ** (-1 / fit[1])
                vals[f"multfit-{key}"] = (f"{eqf / b['params']:.2f}"
                                          + ("$^\\dagger$" if eqf > max(n for n, _ in line) else ""))
            # The fixed part's share of what Chorus adds: all added parameters minus the N-gram rows
            # (the stored tables minus the token table of 8,000 x width).
            if r.get("params_table"):
                added = r["params"] - b["params"]
                fixed = added - (r["params_table"] - 8000 * width[size])
                vals[f"fixedshare-{key}"] = f"{100 * fixed / added:.0f}"
    # The title's trade at each size: the Chorus x1.09 model against a model without a table of its
    # equivalent size, in tokens per second on RTX 5090 cards (b's throughput read off the straight line
    # between neighbouring sizes on logarithmic axes) and in the quiz (b's score at its best rate read off
    # the line between neighbouring sizes on a logarithmic size axis). Only where the equivalent size lies
    # between measured sizes.
    def tok5090(size, arm, tab):
        xs = [r["tok_s"] for r in rs if r["size"] == size and r["arm"] == arm and r["table"] == tab
              and r["tok_s"] and r["card"] and "5090" in r["card"]]
        return statistics.median(xs) if xs else None

    def interp(pts, x, logy):
        pts = sorted(pts)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                f = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
                return math.exp(math.log(y0) + f * (math.log(y1) - math.log(y0))) if logy else y0 + f * (y1 - y0)
        return None
    speed = [(best[(s, "b", "")]["params"], tok5090(s, "b", "")) for s in ORDER if (s, "b", "") in best]
    speed = [(n, t) for n, t in speed if t]
    quizline = [(best[(s, "b", "")]["params"], best[(s, "b", "")]["quiz"]) for s in ORDER
                if (s, "b", "") in best and best[(s, "b", "")]["quiz"] is not None]
    margins = []
    for size in ORDER:
        for tab in ("0.275", "4.35", "max"):
            r, b = best.get((size, "e_chorus", tab)), best.get((size, "b", ""))
            if r and b:
                eq, ext = equivalent(r["loss"], line)
                tb, tc = (interp(speed, eq, True), tok5090(size, "e_chorus", tab)) if eq and not ext else (None, None)
                if tb and tc:
                    vals[f"eqspeed-{size}-chorus{tab}"] = f"{(tc / tb - 1) * 100:.0f}"
        r, b = best.get((size, "e_chorus", "1.09")), best.get((size, "b", ""))
        if not (r and b):
            continue
        eq, ext = equivalent(r["loss"], line)
        if not eq or ext:
            continue
        vals[f"eqsize-{size}"] = f"{eq / 1e6:.0f}M"
        vals[f"params-{size}-chorus1.09"] = f"{r['params'] / 1e6:.0f}M"
        tb, tc = interp(speed, eq, True), tok5090(size, "e_chorus", "1.09")
        if tb and tc:
            vals[f"eqspeed-{size}"] = f"{(tc / tb - 1) * 100:.0f}"
            if r.get("bracketed") is not False:
                margins.append(round((tc / tb - 1) * 100))
        qb = interp(quizline, eq, False)
        if qb is not None:
            vals[f"quizeq-{size}"] = f"{qb:.3f}"
    if margins:
        vals["eqspeed-range"] = f"{min(margins)} to {max(margins)}"
    slow = [-int(vals[f"eqspeed-{s}-chorus0.275"]) for s in ORDER if f"eqspeed-{s}-chorus0.275" in vals]
    if slow:
        vals["eqslow-0.275"] = f"{min(slow)} to {max(slow)}"
    # Training FLOPs per token after Kaplan et al. (2020): 6 per parameter in a matrix product, the tied
    # embedding counted once (as the output head), and 6 n_layer n_ctx d for attention. Chorus's table reads
    # are lookups; its dense parts (gates, convolution, projections: the parameters outside the tables beyond
    # the plain model's) count as matrix products.
    ctx = 1024

    def flops_plain(size, n):
        return 6 * n + 6 * (width[size] // 48) * ctx * width[size]
    plain_flops = [(best[(s, "b", "")]["params"], flops_plain(s, best[(s, "b", "")]["params"])) for s in ORDER
                   if (s, "b", "") in best and s in width]
    flops_add, flops_eq = [], []
    # The cost table: throughput against the plain model of the same size and against a plain model of the
    # same loss (where its equivalent size lies between measured plain sizes), and training FLOPs per token
    # against the same two plain models.
    cost = ["\\begin{tabular}{lllrrrrrr}", "\\toprule",
            " & & & \\multicolumn{4}{c}{Throughput} & \\multicolumn{2}{c}{FLOPs per token} \\\\",
            "\\cmidrule(lr){4-7}\\cmidrule(l){8-9}",
            " & & & \\multicolumn{2}{c}{tokens/s} & \\multicolumn{2}{c}{Chorus relative to} & \\multicolumn{2}{c}{Chorus relative to} \\\\",
            "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}",
            "Size & Model & Card & Chorus & Plain & plain & equal-loss plain & plain & equal-loss plain \\\\", "\\midrule"]
    for size, tab, row, rec in cost_rows:
        key = f"eqspeed-{size}" if tab == "1.09" else f"eqspeed-{size}-chorus{tab}"
        speq = f"{1 + int(vals[key]) / 100:.2f}" if key in vals else ""
        b = best.get((size, "b", ""))
        fl_ratio = fl_eq = ""
        if b and size in width:
            fp = flops_plain(size, b["params"])
            fc = fp + 6 * ((rec["params"] - rec["params_table"]) - b["params"])
            fl_ratio = f"{fc / fp:.2f}"
            r = best.get((size, "e_chorus", tab))
            eq, ext = equivalent(r["loss"], line) if r else (None, True)
            if eq and not ext:
                fe = interp(plain_flops, eq, True)
                if fe:
                    fl_eq = f"{fc / fe:.2f}"
                    if tab == "1.09":
                        flops_eq.append(100 * (1 - fc / fe))
            if tab == "1.09":
                flops_add.append(100 * (fc / fp - 1))
        cost.append(row + f" & {speq} & {fl_ratio} & {fl_eq} \\\\")
    cost += ["\\bottomrule", "\\end{tabular}"]
    for key, xs in (("flopsadd-range", flops_add), ("flopseq-range", flops_eq)):
        if xs:
            lo, hi = f"{min(xs):.0f}", f"{max(xs):.0f}"
            vals[key] = lo if lo == hi else f"{lo} to {hi}"
    (OUT / "tables" / "cost.tex").write_text("\n".join(cost) + "\n")
    # Run-to-run spread of throughput: runs of one configuration (seeds of the same run name) on RTX 5090
    # cards, largest difference relative to the slower run.
    same = defaultdict(list)
    for r in rs:
        if r["tok_s"] and r["card"] and "5090" in r["card"]:
            same[re.sub(r"_s\d+$", "", r["run"])].append(r["tok_s"])
    spreads = [100 * (max(x) - min(x)) / min(x) for x in same.values() if len(x) > 1]
    if spreads:
        vals["tokspread"] = f"{max(spreads):.0f}"
    # Verbatim overlap of the scored held-out windows with the training split
    # (scripts/heldout_overlap.py) and Chorus's gain by overlap, with window-bootstrap intervals
    # (scripts/overlap_gain.py); both run on mlbox's CPU, results copied to metrics/.
    ov = load(ROOT / "metrics" / "heldout_overlap_2026-11-24.json")
    if ov:
        for n, e in ov["per_n"].items():
            vals[f"ov{n}-share"] = f"{100 * e['share_of_scored_ngrams_found']:.1f}"
            vals[f"ov{n}-any"] = f"{e['windows_with_any']:,}".replace(",", "{,}")
            vals[f"ov{n}-over10"] = f"{e['windows_over_10pct']:,}".replace(",", "{,}")
            vals[f"ov{n}-over50"] = f"{e['windows_over_50pct']:,}".replace(",", "{,}")
            vals[f"ov{n}-over90"] = f"{e['windows_over_90pct']:,}".replace(",", "{,}")
            vals[f"ov{n}-checked"] = f"{e['match_check']['equal']} of {e['match_check']['checked']}"
    og = load(ROOT / "metrics" / "overlap_gain_2026-11-25.json")
    if og:
        for k, v in og["bins"].items():
            vals[f"ogwin-{k.replace('_', '')}"] = f"{v:,}".replace(",", "{,}")
        for size, e in og["sizes"].items():
            for k, g in e["gain"].items():
                vals[f"oggain-{size}-{k.replace('_', '')}"] = f"{g:.3f}"
            lo, hi = e["gain_ci95"]
            vals[f"ogci-{size}"] = f"{lo:.3f} to {hi:.3f}"
            lo, hi = e["multiplier_ci95"]
            vals[f"ogmult-{size}"] = f"{e['multiplier']:.2f}" + ("$^\\dagger$" if e.get("extrapolated") else "")
            vals[f"ogmultci-{size}"] = f"{lo:.3f} to {hi:.3f}"
            for k in ("50gram_none", "13gram_none"):
                if f"multiplier_{k}" in e:
                    vals[f"ogmult-{size}-{k.replace('_', '')}"] = (f"{e[f'multiplier_{k}']:.2f}"
                                                                 + ("$^\\dagger$" if e[f"extrapolated_{k}"] else ""))
        for size, e in og["sizes"].items():
            for k in ("50gram_none", "13gram_none"):
                if e["gain"].get("all") and k in e["gain"]:
                    vals[f"ogkeep-{size}-{k.replace('_', '')}"] = f"{100 * e['gain'][k] / e['gain']['all']:.0f}"
        keep = [100 * e["gain"]["50gram_none"] / e["gain"]["all"] for e in og["sizes"].values()
                if e["gain"].get("all") and "50gram_none" in e["gain"]]
        if keep:
            vals["ogkeep50"] = f"{min(keep):.0f} to {max(keep):.0f}"
    # Multiplier of equal-size tables against backbone size in three settings, as pgfplots commands;
    # points read beyond the largest plain model are open marks and are not joined.
    msize = []
    series = [("mult-{}-chorus1.09-lr1", "black!45, thick, dashed", "*", "base rate, tables' rate 0.0015"),
              ("mult-{}-chorus1.09-best", "black, thick, solid", "square*", "best backbone rates, tables' rate 0.0015"),
              ("trmult-{}-tr0p33", "blue!70!black, thick, dashdotted", "triangle*",
               "best backbone rates, tables' rate 0.0005")]
    for key, style, mark, label in series:
        meas, extra = [], []
        for s in ORDER:
            v = vals.get(key.format(s))
            if v is None or (s, "b", "") not in best:
                continue
            pt = f"({best[(s, 'b', '')]['params'] / 1e6:.1f},{float(v.split('$')[0]):.2f})"
            (extra if "dagger" in v else meas).append(pt)
        msize.append(f"\\addplot[{style}, mark={mark}, mark options={{solid}}] coordinates {{{' '.join(meas)}}};")
        msize.append(f"\\addlegendentry{{{label}}}")
        if extra:
            colour = style.split(",")[0]
            open_mark = "o" if mark == "*" else mark.rstrip("*")
            msize.append(f"\\addplot[{colour}, only marks, mark={open_mark}, mark options={{solid, fill=white}}, "
                         f"forget plot] coordinates {{{' '.join(extra)}}};")
    (OUT / "tables" / "mult_size.tex").write_text("\n".join(msize) + "\n")
    body = "".join(f"\\expandafter\\def\\csname v@{k}\\endcsname{{{v}}}\n" for k, v in sorted(vals.items()))
    (OUT / "numbers.tex").write_text(
        "% Generated by scripts/paper_numbers.py from metrics/; do not edit.\n"
        "% Use \\val{key}, e.g. \\val{mult-47M-chorus1.09-lr1}; a missing key prints a red mark.\n"
        "\\newcommand{\\val}[1]{\\ifcsname v@#1\\endcsname\\csname v@#1\\endcsname\\else\\textcolor{red}{[#1]}\\fi}\n"
        + body)
    print(f"{len(rs)} runs, {len(c)} cells")


if __name__ == "__main__":
    main()
