# Prompt selection

`protocol.json` is the protocol frozen before the selection runs: fixed candidate pool, support size,
presentation budget and training seed per setting; support seed and training seed recorded separately.

## Methods

| Method | Rule | Script |
|---|---|---|
| Uniform random | PCG64 permutation of the candidate pool, nested prefixes (M = 8 ⊂ 48) | `select_uniform.py` |
| Stratified random | proportional source and prompt-length strata (+ frozen pre-training difficulty where available), random within strata | `rank_stratified.py` |
| Semantic diversity | normalized Qwen3-Embedding-0.6B prompt embeddings, deterministic farthest-first traversal starting at the candidate nearest the centroid | `rank_semantic.py` |
| Hard | lowest Initial-student pass rate under a frozen K = 8 rollout per candidate, hash tie-break | `rank_hard.py` |
| Shortest | lowest mean Initial-student response length under the same frozen rollout | `rank_shortest.py` |
| Cost-aware D-opt | reliability-aware gradient sketches + cost-aware greedy D-optimal design | `cost_d_opt/select_cost_d_opt.py` (with the helper modules in `cost_d_opt/`) |

Ranked methods are turned into nested supports and balanced schedules by
`select_ranked.py`. `update_representativeness.py` computes the
update-representativeness error E(S) of a support against the pool-mean initial gradient.

Not included: the large pre-computation jobs (Initial-student rollouts for hard/shortest, the embedding
pass for semantic, gradient sketches for D-opt). The supports they produced are shipped below.

## Settings, supports and runs

Each arm is a run config `configs/selection/<pair>/<run>.yaml`; its training data is
`data/schedules/selection/<pair>/<run>/` (prompt-ID tables + inline prompts), and
`data/schedules/INDEX.csv` lists method and seeds. Run names are `<method>_m<M>[_sel<support seed>]_s<steps>`;
all arms use training seed 42. Candidate pools: `data/selection/candidate_pools/`.

**JustRL-DeepSeek-1.5B → DeepSeek-R1-Distill-Qwen-1.5B** (`justrl_1p5b_to_r1_distill_1p5b`), DeepMath,
C = 384 (`justrl_deepmath_c384`), M = 8, 65 steps × 256:
`uniform_m8_sel{42,43,44}_s65`, `stratified_m8_sel42_s65`, `semantic_m8_sel42_s65`, `hard_m8_sel42_s65`,
`shortest_m8_s65`, `cost_d_opt_m8_s65`.

**Code GRPO-300 → Qwen3-4B** (`code_grpo300_to_qwen3_4b`), Eurus code, C = 384 (`eurus_code_c384`), M = 48,
100 steps × 256: `uniform_m48_sel42_s100`, `semantic_m48_sel42_s100`, `hard_m48_sel42_s100`.

**Qwen3-30B-A3B-Instruct-2507 → Qwen3-4B** (`qwen3_30b_to_qwen3_4b`), DeepMath L6, C = 2304
(`qwen3_30b_deepmath_c2304`), M = 48, 25 steps × 256: `uniform_m48_sel{42,43,44}_s25`,
`stratified_m48_sel{42,43,44}_s25`, `semantic_m48_sel42_s25`, `hard_m48_sel42_s25`, `shortest_m48_sel42_s25`.

Each `support.csv` lists the selected prompts (item hash, prompt hash, source dataset index) in selection
order; `order.csv.gz` gives the training order.

The scripts keep their original defaults, which refer to the research repository layout; set
`OPD_PROJECT_ROOT` to a directory with that layout or pass paths explicitly.
