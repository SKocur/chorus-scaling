"""The fact quiz of ../wieszcz-xix-chat for checkpoints of this track.

`fact_quiz.py` there reads a prompt once and carries only the attention cache into the
greedy continuation and the scoring of candidates. A model with an Engram module also has
to carry the last tokens and the convolution's left context; without them the first
continued tokens are looked up with padded N-grams. This asks the same questions with the
same scoring and carries that state too. It takes checkpoints with or without the module,
so both arms of a comparison go through one script.

    PYTHONPATH=../wieszcz-xix/src:src python src/fact_quiz_engram.py \
        --checkpoint <final.pt> --tag e_47m --out metrics/fact_quiz_e_47m_<date>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import ByteLevelBPETokenizer

from engram import EngramState, load_model, quantize_table, zero_table
from train import get_device

REPO = Path(__file__).resolve().parent.parent
QUIZ_DIR = Path(os.environ.get("WIESZCZ_CHAT_SCRIPTS", REPO.parent / "wieszcz-xix-chat" / "scripts"))
sys.path.insert(0, str(QUIZ_DIR))
from fact_quiz import Asker, ask_as_completion, summarise  # noqa: E402
from quiz_facts import RANKED, facts  # noqa: E402


class EngramAsker(Asker):
    def read(self, ids: list[int]):
        caches = [self.cache_type() for _ in self.model.blocks]
        logits, _ = self.model(torch.tensor([ids], device=self.device), caches=caches)
        return logits[0, -1], [(c.k, c.v, getattr(c, "engram", None)) for c in caches]

    def resume(self, state, rows: int):
        caches = super().resume([(k, v) for k, v, _ in state], rows)
        for cache, (_, _, read) in zip(caches, state):
            if read is not None:
                cache.engram = EngramState()
                cache.engram.tokens = read.tokens.expand(rows, -1)
                cache.engram.conv_tail = read.conv_tail.expand(rows, -1, -1)
        return caches


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-dir", default=str(REPO.parent / "wieszcz-xix" / "tokenizer"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--groups", nargs="*", default=[], help="only these groups")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--zero-table", nargs="+", default=None, metavar="ORDER",
                    help="zero the Engram table rows of these N-gram orders, or 'all'")
    ap.add_argument("--table-bits", type=int, default=None,
                    help="round the Engram table to this many bits per value, one scale per row")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    device = args.device or get_device()
    tok_dir = Path(args.tokenizer_dir)
    tokenizer = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))
    model, cfg, step, n_params = load_model(args.checkpoint, device)
    zeroed = None
    if args.zero_table:
        zeroed = zero_table(model, None if args.zero_table == ["all"] else [int(n) for n in args.zero_table])
        print(f"zeroed {zeroed} table rows", flush=True)
    quantized = quantize_table(model, args.table_bits) if args.table_bits else None
    asker = EngramAsker(model, tokenizer, device)

    groups: dict[str, list[dict]] = defaultdict(list)
    for fact in facts():
        if not args.groups or fact["group"] in args.groups:
            groups[fact["group"]].append(fact)

    started = time.time()
    with torch.inference_mode():
        rows = ask_as_completion(asker, groups, args.shots, args.seed, args.max_new_tokens)
    summary = summarise(rows)
    report = {
        "meta": {
            "script": "src/fact_quiz_engram.py", "script_sha256": sha16(Path(__file__)),
            "quiz_sha256": {f: sha16(QUIZ_DIR / f) for f in ("fact_quiz.py", "quiz_facts.py")},
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(), "device": device,
        },
        "args": vars(args), "model": args.tag, "step": step, "engram": cfg.get("engram"),
        "zeroed_table_rows": zeroed, "quantized_table": quantized,
        "candidates": {name: sorted({f["answer"] for f in group})
                       for name, group in groups.items() if name in RANKED},
        "parameters": n_params, "seconds": round(time.time() - started, 1),
        "summary": summary, "rows": rows,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, entry in summary.items():
        print(f"{args.tag:>22} {name:>24} " + "  ".join(f"{k} {v}" for k, v in entry.items()))


if __name__ == "__main__":
    main()
