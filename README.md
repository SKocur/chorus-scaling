# Bigger Model or Bigger Table? Code and results

Code, configurations and per-run results of the preprint "Bigger Model or Bigger Table? How Much
Backbone a Chorus N-gram Module Replaces" (Szymon Kocur, 2026). Every number in the paper is computed
from the files in `metrics/` by `scripts/paper_numbers.py`.

| Path | What |
|---|---|
| `src/` | training (`train_engram.py`), the Chorus module (`chorus.py`), shared table code and model loading (`engram.py`), the Ember table optimiser (`ember.py`), held-out loss (`eval_engram.py`), bits per byte (`eval_bpb.py`), the fact quiz (`fact_quiz_engram.py`) |
| `quiz/` | the fact quiz's questions (`quiz_facts.py`) and scoring (`fact_quiz.py`) |
| `configs/` | the configuration of every run in the paper |
| `metrics/screen/<run>/` | per run: held-out loss (`eval_final.json`), bits per byte (`eval_bpb.json`, where measured), the manifest (configuration, parameter counts, code and data hashes, card), training and validation curves with throughput (`*_train.csv`, `*_val.csv`) |
| `metrics/fact_quiz*/` | quiz results per run, with the N-gram tables zeroed in `*_zero_all.json` |
| `metrics/quant_chorus/` | held-out loss and quiz with the tables stored in fewer bits |
| `metrics/heldout_overlap_*.json`, `metrics/overlap_gain_*.json` | overlap of the held-out windows with the training split, and Chorus's gain by overlap |
| `scripts/` | `paper_numbers.py` (all numbers and tables of the paper), `heldout_overlap.py`, `overlap_gain.py` |

## Reproducing the paper's numbers

    python3 scripts/paper_numbers.py      # writes paper/numbers.tex and paper/tables/*.tex

`paper_numbers.py` needs only the Python standard library.

## Backbone and data

The backbone (`model.py`, `train.py`, `eval_val.py`, `metrics_logger.py`) is imported from
[SKocur/wieszcz-xix](https://github.com/SKocur/wieszcz-xix) at commit `c885113`; every run's
manifest records the SHA-256 of the `model.py` and `train.py` it used. The training and held-out
token streams (`tokens_frozen_6.69B.bin`, `val_2026-08-03.bin`, uint16, the corpus's 8,000-token
tokenizer) are built from the [Wieszcz-XIX corpus](https://huggingface.co/datasets/SKocur/polish-pre1918-corpus)
as that repository's README describes. Runs used PyTorch 2.13 (CUDA 13.0) with Python 3.12 on
RTX 5090 cards and Python 3.14 on an RTX 3080.

    git clone https://github.com/SKocur/wieszcz-xix && git -C wieszcz-xix checkout c885113
    export PYTHONPATH=wieszcz-xix/src:src

## Commands

Training (each run, then its evaluation on 8,192 held-out windows):

    python src/train_engram.py --config configs/screen_107m/e_chorus_107m_lr8.json --seed 1337 \
        --run screen_e_chorus_107m_lr8_s1337 --data tokens_frozen_6.69B.bin --val-data val_2026-08-03.bin \
        --ckpt-dir checkpoints/screen_e_chorus_107m_lr8_s1337 --metrics-dir runs/screen_e_chorus_107m_lr8_s1337
    python src/eval_engram.py --ckpt checkpoints/screen_e_chorus_107m_lr8_s1337/final.pt \
        --data val_2026-08-03.bin --windows 8192 --val-source-boundary 55758336 --out eval_final.json

Half and twice the budget resume a full run's saved state with `--resume <state> --max-steps <n>`:
half the steps (2,552) from the full run's state at step 2,297 (`stable_branch.pt`, written at the
config's `branch_ckpt_step`), twice the steps (10,208) from its state at step 4,594
(`predecay.pt`). The 47M half runs train from step 0 with `--max-steps 2552`.

Fact quiz (CPU is enough):

    WIESZCZ_CHAT_SCRIPTS=quiz python src/fact_quiz_engram.py --checkpoint <final.pt> --tag <run> \
        --device cpu --tokenizer-dir <tokenizer directory> --out metrics/fact_quiz/<run>.json
    # add --zero-table all for the quiz with the N-gram tables set to zero

## Run names

`screen_<arm>[_<table>][_<size>][_lr<k>][_mb<m>][_<budget>]_s<seed>`:

- arm: `b` the plain model, `e_chorus` Chorus, `b_ve` Chorus's fixed part only;
- table: none for equal-size tables (added ratio 1.09), `small` 0.275, `big` 4.35, `huge` the largest;
- size: none for 47M, `16m`, `107m`, `203m` (204M), `349m`, `550m`;
- `lr<k>`: Muon learning rate k x 0.015 (none: the base rate 0.015; `lr0p5` is 0.0075);
- `mb<m>`: the same tokens per step in micro-batches of m sequences;
- budget: `half` (resumed), `halffresh` (from step 0), `double`; none for the full budget of 5,104 steps.

Side runs: the tables' own rate is 0.0015 unless a tag changes it. `tr0p33` sets it to 0.0005 at every
size, `tr0p11` to 0.00017 at 16M and 47M, `tr0p04` to 0.000056 at 16M, and `tr3` and `lrmult*` change it
at 47M. `ember*` train the tables with Ember and `32k` and `48k` use larger vocabularies, both at 47M.

Checkpoints are not included.

## Third-party code

See `NOTICE`.
