# Data

## `schedules/` — exact training data of every run

verl reads the training parquet in file order (`data.shuffle=False`), so row *i* is prompt *i mod B* of
optimizer step *i // B*. Every training file that a released run consumed is stored as a prompt-ID table
(`data_prep/schedule_ids.py`), one directory per run.
`schedules/INDEX.csv` lists every file with its number of distinct prompts, rows used, batch, steps, data
source, selection method, selection seed and training seed. Each directory contains:

| File | Content |
|---|---|
| `order.csv.gz` | one row per training presentation: `position, step, item_sha256, prompt_sha256, data_source, source_index, source_split, extra_info_delta` |
| `support.csv` | the distinct prompts in first-appearance order with their presentation counts |
| `support.parquet` | those prompts as full verl rows (messages, answer/tests, `extra_info`); present when small enough |
| `support_extra_info.jsonl.gz` | when `support.parquet` is absent: first-appearance `extra_info` of each prompt |

`item_sha256` is the sha256 of the canonical JSON of the row without `extra_info` (prompt, answer or tests,
data source); `prompt_sha256` covers the chat messages only. `source_index` is the row index in the source
dataset (`extra_info.index`, or `sample_id` for Open-R1 code). The per-presentation `extra_info` carries
the selection method and seeds of the materializer.

Rebuild a parquet (done automatically by `opd/run.py`):

```bash
python data_prep/schedule_ids.py rebuild data/schedules/<path> \
    --rows data/schedules/<path>/support.parquet [<pool.parquet> ...] --out schedule.parquet
```

The four schedules without `support.parquet` (Qwen3-4B Eurus code M = 3840 and full, Qwen3-1.7B Open-R1
code M = 3840 and full) take their rows from the Eurus pool (`$POOL_ROOT/eurus_code.parquet`)
and `pools/openr1_code.parquet`. Every rebuild was checked row by row against the original file;
the rebuilt file's own sha256 differs because parquet encoding/compression differ.

### Layout, nesting and pairing

`schedules/<group>/<pair>/<run>/` holds the training data of `configs/<group>/<pair>/<run>.yaml`
(resumed runs point to the data of the run they continue).

- M-sweeps use nested supports: the M = 1 prompt is contained in M = 8/48/..., drawn with the selection seed
  listed in `INDEX.csv`. The DeepMath supports of `math_grpo500_to_qwen3_4b` (`deepmath_m*_s15`) are the same
  prompts and order as `qwen3_30b_to_qwen3_4b` and `qwen3_30b_to_qwen3_1p7b`; `justrl_1p5b_to_r1_distill_1p5b`
  uses the same M = 1/48/3840 prompts with a 100-step schedule.
- Teacher switch (`teacher_switch/math_or_code_grpo_to_qwen3_4b/`): for each `sel<S>_train<T>` block the
  `math_teacher_*` and `code_teacher_*` runs have byte-identical training data, so both teachers see the same
  48 DeepMath or 48 Eurus-code prompts in the same order. Support seeds 43/44 draw disjoint supports with the
  same nested stratified rule as seed 42; training seeds 43/44 keep the seed-42 prompts and only change the
  presentation order and verl's RNG seeds.

## `pools/` — source prompt pools

| File | Rows | Source |
|---|---|---|
| `deepmath_l6.parquet` | 57,046 | DeepMath-103K, difficulty ≥ 6 (G-OPD training split) |
| `dapo_clean.parquet` | 17,170 | DAPO-Math-17k deduplicated, answer conflicts and AIME/HMMT overlaps removed (`data_prep/build_dapo_clean_pool.py`) |
| `textbook_science.parquet` | 126,397 | MegaScience/TextbookReasoning, science subjects, question-only, image/context-dependent items removed |
| `openr1_code.parquet` | 6,519 | open-r1/verifiable-coding-problems-python@b761a24, reference-validated and screened |
| not shipped: `eurus_code.parquet` | 22,618 | PRIME-RL/Eurus-2-RL-Data@9776b13, stdio code tests (`data_prep/build_eurus_code_pool.py`) |

`eurus_code.parquet` (830 MB) is rebuilt with `python data_prep/build_eurus_code_pool.py`, which downloads the
pinned upstream snapshot and writes `data/pools/eurus_code.parquet` (the default `$POOL_ROOT`); the result is
row by row identical to the file used for OPD and for the Code GRPO-300 teacher. The SFT data of the
Qwen3-1.7B-Base student are built by `data_prep/build_sft_code_data.py` (see `models/teacher_training/`).

GSM8K (openai/gsm8k main train, 7,473), Light-R1 stage2 (qihoo360/Light-R1-SFTData, 3,533 questions) and
LLM-Fusion IF/Agent prompts appear only through M ≤ 48 supports (or the full Light stage2 set), which are
stored inline.

## Other

- `selection/candidate_pools/`: candidate pools of the selection benchmark (ID tables).
- `probes/`: frozen probe cohorts for the functional analysis (`analysis/README.md`).
- `val/`: validation sets monitored during training (AIME24/25, Eurus code dev, Open-R1 dev, IFEval, BFCL-v3).
- `eval/math/`: AIME24, AIME25, HMMT25-Feb, HMMT25-Nov test sets (30 each, `SHA256SUMS`);
  `eval/gpqa/`: GPQA-Diamond.

## `data_prep/`

Materializers that produced the frozen supports and schedules (nested stratified draws, schedule balancing):
DeepMath (`deepmath_nested_supports.py`, `deepmath_schedules_s100.py`,
`deepmath_m48_schedule_s60.py`), DAPO (`dapo_nested_supports.py`), Eurus code
(`eurus_code_nested_supports.py`), Open-R1 code (`openr1_code_nested_supports.py`),
teacher-switch grid (`teacher_switch_schedules.py`). They refuse to overwrite an existing
freeze and resolve their defaults against `OPD_PROJECT_ROOT` (a directory with the research-repository
layout); the frozen outputs above are what the runs used.
