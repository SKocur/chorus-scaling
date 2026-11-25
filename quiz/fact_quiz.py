"""Put the facts of `quiz_facts.py` to a model and score what it knows.

A base model is asked in two ways, neither of which draws a sample:

completion  the fact opens a sentence and the model continues it greedily. The answer is
            right when the first clause of the continuation matches the fact's pattern.
ranking     in the groups whose answers are of one kind, every answer of the group is
            scored as the continuation and the right one is ranked among them. This does
            not depend on how the model words its answer. The scores are kept, so
            that `fact_quiz_compare.py` can take out what a well-known answer scores
            whatever is asked.

Nothing is put before the sentence. With `--shots` a few other facts of the group can be
stated first, drawn with a seed that does not depend on the model, but that was tried and
is not the default: at three stated facts a quarter of the 349M model's continuations
repeated one of their answers, and its ranking of the right year fell from 67% to 36%.

With `--chat` the fact is asked as a question in the conversation format instead, and the
greedy answer is matched against the pattern anywhere in it. That is the more lenient
test, and it is the one a chat model meets in use. Adding `--open-answers` writes the
sentence opening as the first words of the answer, which separates what the model fails to
recall from what it fails to say because of how it begins.

    ../wieszcz-xix/.venv/bin/python scripts/fact_quiz.py \
        --checkpoint ../models/wieszcz-349m-final/model.pt --tag 349m_final \
        --out metrics/fact_quiz_349m_final_2026-11-13.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

from chat_format import EOT, encode_prompt, load_model, load_tokenizer
from common import run_meta
from quiz_facts import RANKED, facts, key

# A clause ends at a stop after a word of four letters or a digit, at a comma after three,
# or at a line break, so that "ks." and an initial do not cut the answer short.
CLAUSE_END = re.compile(r"(?:(?<=\w{4})|(?<=\d))[.;:?!]|(?<=\w{3}),|\n")


def shots_for(fact: dict, frame: int, group: list[dict], k: int, seed: int) -> list[dict]:
    rng = random.Random(f"{seed}:{fact['group']}:{fact['index']}:{frame}")
    accepts = key(fact)
    others = [f for f in group if f is not fact]
    rng.shuffle(others)
    chosen: list[dict] = []
    for other in others:
        if len(chosen) == k:
            break
        sentence = f"{other['prompts'][frame]} {other['answer']}."
        if accepts.search(sentence) or any(other["answer"] == c["answer"] for c in chosen):
            continue
        chosen.append(other)
    return chosen


def first_clause(text: str) -> str:
    """The continuation up to its first clause end; a name set in capitals is put in ordinary case."""
    text = text.lstrip()
    end = CLAUSE_END.search(text)
    clause = text[:end.start()] if end else text
    return clause.title() if clause.isupper() else clause


class Asker:
    def __init__(self, model, tokenizer, device: str) -> None:
        from model import KVCache

        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.cache_type = KVCache
        self.eot = tokenizer.token_to_id(EOT)

    def read(self, ids: list[int]):
        """Next-token scores after `ids`, and the attention cache that produced them."""
        caches = [self.cache_type() for _ in self.model.blocks]
        logits, _ = self.model(torch.tensor([ids], device=self.device), caches=caches)
        return logits[0, -1], [(c.k, c.v) for c in caches]

    def resume(self, state, rows: int):
        caches = []
        for k, v in state:
            cache = self.cache_type()
            cache.k, cache.v = k.expand(rows, -1, -1, -1), v.expand(rows, -1, -1, -1)
            caches.append(cache)
        return caches

    def continue_greedily(self, ids: list[int], last, state, max_new: int, done) -> str:
        caches = self.resume(state, 1)
        produced: list[int] = []
        for step in range(max_new):
            token = int(last.argmax())
            if token == self.eot:
                break
            produced.append(token)
            if done(self.tokenizer.decode(produced)):
                break
            logits, _ = self.model(torch.tensor([[token]], device=self.device), caches=caches,
                                   start_pos=len(ids) + step)
            last = logits[0, -1]
        return self.tokenizer.decode(produced)

    def score(self, ids: list[int], last, state, candidates: list[str]) -> list[float]:
        """Log-probability of each candidate as the continuation of `ids`."""
        pieces = [self.tokenizer.encode(f" {c}").ids for c in candidates]
        first = torch.log_softmax(last.float(), dim=-1)
        totals = [float(first[p[0]]) for p in pieces]
        caches = self.resume(state, len(pieces))
        for step in range(max(len(p) for p in pieces) - 1):
            fed = torch.tensor([[p[step] if step < len(p) else self.eot] for p in pieces],
                               device=self.device)
            logits, _ = self.model(fed, caches=caches, start_pos=len(ids) + step)
            scores = torch.log_softmax(logits[:, -1].float(), dim=-1)
            for row, piece in enumerate(pieces):
                if step + 1 < len(piece):
                    totals[row] += float(scores[row, piece[step + 1]])
        return totals


def ask_as_completion(asker: Asker, groups: dict[str, list[dict]], shots: int, seed: int,
                      max_new: int) -> list[dict]:
    rows = []
    for name, group in groups.items():
        answers = sorted({f["answer"] for f in group})
        for fact in group:
            for frame, opening in enumerate(fact["prompts"]):
                shown = shots_for(fact, frame, group, shots, seed)
                text = " ".join([f"{s['prompts'][frame]} {s['answer']}." for s in shown]
                                + [opening])
                ids = [asker.eot] + asker.tokenizer.encode(text).ids
                last, state = asker.read(ids)
                said = asker.continue_greedily(
                    ids, last, state, max_new, lambda t: bool(CLAUSE_END.search(t.lstrip())))
                row = {"group": name, "index": fact["index"], "frame": frame,
                       "asked": opening, "answer": fact["answer"], "said": said.strip(),
                       "right": bool(key(fact).search(first_clause(said))),
                       "shown": [s["answer"] for s in shown]}
                if name in RANKED:
                    taken = {s["answer"] for s in shown}
                    candidates = [a for a in answers if a not in taken]
                    scores = asker.score(ids, last, state, candidates)
                    own = scores[candidates.index(fact["answer"])]
                    row["rank"] = 1 + sum(s > own for s in scores)
                    row["candidates"] = len(candidates)
                    row["preferred"] = candidates[max(range(len(scores)), key=scores.__getitem__)]
                    if not shown:
                        row["scores"] = [round(s, 3) for s in scores]
                rows.append(row)
    return rows


def ask_as_question(asker: Asker, groups: dict[str, list[dict]], max_new: int,
                    opened: bool) -> list[dict]:
    rows = []
    for name, group in groups.items():
        for fact in group:
            ids = encode_prompt(asker.tokenizer, [], fact["question"])
            if opened:
                ids += asker.tokenizer.encode(f" {fact['prompts'][0]}").ids
            last, state = asker.read(ids)
            said = asker.continue_greedily(ids, last, state, max_new, lambda t: "\n" in t)
            said = said.split("\n")[0].strip()
            rows.append({"group": name, "index": fact["index"], "frame": 0,
                         "asked": fact["question"], "answer": fact["answer"], "said": said,
                         "right": bool(key(fact).search(said))})
    return rows


def summarise(rows: list[dict]) -> dict:
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    out = {}
    for name, part in by_group.items():
        entry = {"facts": len({r["index"] for r in part}), "asked": len(part),
                 "right_share": round(sum(r["right"] for r in part) / len(part), 4)}
        ranked = [r for r in part if "rank" in r]
        if ranked:
            entry["ranked_first_share"] = round(sum(r["rank"] == 1 for r in ranked) / len(ranked), 4)
            entry["mean_reciprocal_rank"] = round(sum(1 / r["rank"] for r in ranked) / len(ranked), 4)
            entry["mean_candidates"] = round(sum(r["candidates"] for r in ranked) / len(ranked), 1)
        out[name] = entry
    for measure in ("right_share", "ranked_first_share", "mean_reciprocal_rank"):
        values = [entry[measure] for entry in out.values() if measure in entry]
        if values:
            out.setdefault("mean_over_groups", {})[measure] = round(sum(values) / len(values), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chat", action="store_true",
                    help="ask each fact as a question in the conversation format")
    ap.add_argument("--open-answers", action="store_true",
                    help="with --chat, begin each answer with the fact's sentence opening, to "
                         "tell what the model knows from how it chooses to begin")
    ap.add_argument("--shots", type=int, default=0,
                    help="facts of the same group stated in full before the one asked")
    ap.add_argument("--groups", nargs="*", default=[], help="only these groups")
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="default 16 for a completion, 60 for an answer to a question")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = load_tokenizer()
    model, cfg = load_model(args.checkpoint, device)
    model.eval()
    asker = Asker(model, tokenizer, device)

    groups: dict[str, list[dict]] = defaultdict(list)
    for fact in facts():
        if not args.groups or fact["group"] in args.groups:
            groups[fact["group"]].append(fact)

    started = time.time()
    with torch.inference_mode():
        if args.chat:
            rows = ask_as_question(asker, groups, args.max_new_tokens or 60, args.open_answers)
        else:
            rows = ask_as_completion(asker, groups, args.shots, args.seed,
                                     args.max_new_tokens or 16)
    summary = summarise(rows)
    candidates = {name: sorted({f["answer"] for f in group}) for name, group in groups.items()
                  if name in RANKED}
    report = {"meta": run_meta(__file__), "args": vars(args), "model": args.tag,
              "candidates": candidates,
              "parameters": sum(p.numel() for p in model.parameters()),
              "seconds": round(time.time() - started, 1), "summary": summary, "rows": rows}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, entry in summary.items():
        print(f"{args.tag:>22} {name:>24} " + "  ".join(f"{k} {v}" for k, v in entry.items()))


if __name__ == "__main__":
    main()
