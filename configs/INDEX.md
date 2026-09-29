# Run index

`M` is the number of distinct prompts the run
trained on; `status` is how the original run ended.

## selection / code_grpo300_to_qwen3_4b

student `qwen3_4b`; teacher `code_grpo300_qwen3_4b`

| run | M | steps | batch | lr | seed | status |
|---|---|---|---|---|---|---|
| `hard_m48_sel42_s100` | 48 | 100 | 256 | 1e-05 | 42 | completed |
| `semantic_m48_sel42_s100` | 48 | 100 | 256 | 1e-05 | 42 | completed |
| `uniform_m48_sel42_s100` | 48 | 100 | 256 | 1e-05 | 42 | completed |

## selection / justrl_1p5b_to_r1_distill_1p5b

student `r1_distill_qwen_1p5b`; teacher `justrl_deepseek_1p5b`

| run | M | steps | batch | lr | seed | status |
|---|---|---|---|---|---|---|
| `cost_d_opt_m8_s65` | 8 | 65 | 256 | 1e-05 | 42 | completed |
| `hard_m8_sel42_s65` | 8 | 65 | 256 | 1e-05 | 42 | completed |
| `semantic_m8_sel42_s65` | 8 | 65 | 256 | 1e-05 | 42 | completed |
| `shortest_m8_s65` | 8 | 65 | 256 | 1e-05 | 42 | completed |
| `stratified_m8_sel42_s65` | 8 | 65 | 256 | 1e-05 | 42 | completed |
| `uniform_m8_sel42_s65` | 8 | 65 | 256 | 1e-05 | 42 | completed |
| `uniform_m8_sel43_s65` | 8 | 65 | 256 | 1e-05 | 42 | completed |
| `uniform_m8_sel44_s65` | 8 | 65 | 256 | 1e-05 | 42 | completed |

## selection / qwen3_30b_to_qwen3_4b

student `qwen3_4b`; teacher `qwen3_30b_a3b_instruct_2507`

| run | M | steps | batch | lr | seed | status |
|---|---|---|---|---|---|---|
| `hard_m48_sel42_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `semantic_m48_sel42_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `shortest_m48_sel42_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `stratified_m48_sel42_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `stratified_m48_sel43_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `stratified_m48_sel44_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `uniform_m48_sel42_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `uniform_m48_sel43_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `uniform_m48_sel44_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |

## support_size / code_grpo300_to_qwen3_4b

student `qwen3_4b`; teacher `code_grpo300_qwen3_4b`

| run | M | steps | batch | lr | seed | status |
|---|---|---|---|---|---|---|
| `eurus_code_full_s100` | 22618 | 100 | 256 | 1e-05 | 42 | completed |
| `eurus_code_m1_s100` | 1 | 100 | 256 | 1e-05 | 42 | completed |
| `eurus_code_m3840_s100` | 3840 | 100 | 256 | 1e-05 | 42 | completed |
| `eurus_code_m48_s100` | 48 | 100 | 256 | 1e-05 | 42 | completed |

## support_size / code_grpo400_to_qwen3_1p7b_sft

student `qwen3_1p7b_base_sft554`; teacher `code_grpo400_qwen3_1p7b`

| run | M | steps | batch | lr | seed | status |
|---|---|---|---|---|---|---|
| `openr1_code_full_s50` | 6519 | 50 | 256 | 1e-05 | 42 | completed |
| `openr1_code_m1_s100_resume50` | 1 | 100 | 256 | 1e-05 | 42 | completed |
| `openr1_code_m1_s50` | 1 | 50 | 256 | 1e-05 | 42 | completed |
| `openr1_code_m256_s100_resume50` | 256 | 100 | 256 | 1e-05 | 42 | completed |
| `openr1_code_m256_s50` | 256 | 50 | 256 | 1e-05 | 42 | completed |
| `openr1_code_m3840_s50` | 3840 | 50 | 256 | 1e-05 | 42 | completed |

## support_size / deepscaler_1p5b_to_r1_distill_1p5b

student `r1_distill_qwen_1p5b`; teacher `deepscaler_1p5b_preview`

| run | M | steps | batch | lr | seed | status |
|---|---|---|---|---|---|---|
| `dapo_m8_s100` | 8 | 100 | 256 | 1e-05 | 42 | completed |
| `dapo_m8_s100_resume50` | 8 | 100 | 256 | 1e-05 | 42 | stopped at step 65 |
| `deepmath_m8_s100` | 8 | 100 | 256 | 1e-05 | 42 | completed |
| `deepmath_m8_s100_resume50` | 8 | 100 | 256 | 1e-05 | 42 | stopped at step 75 |
| `deepmath_m8_s50_resume25` | 8 | 50 | 256 | 1e-05 | 42 | completed |

## support_size / inst_mix_to_qwen3_4b_instruct

student `qwen3_4b_instruct_2507`; teacher `qwen3_4b_inst_mix`

| run | M | steps | batch | lr | seed | status |
|---|---|---|---|---|---|---|
| `agent_m48_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `deepmath_m48_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `eurus_code_m48_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `eurus_code_m48_sel43_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `eurus_code_m48_sel44_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `if_m48_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `mix3_m48_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `mix5_m48_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `textbook_science_m48_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |

## support_size / justrl_1p5b_to_r1_distill_1p5b

student `r1_distill_qwen_1p5b`; teacher `justrl_deepseek_1p5b`

| run | M | steps | batch | lr | seed | status |
|---|---|---|---|---|---|---|
| `dapo_full_s100` | full pool (shuffled) | 100 | 256 | 1e-05 | 42 | stopped at step 50 |
| `dapo_m1_s100` | 1 | 100 | 256 | 1e-05 | 42 | completed |
| `dapo_m2_s100` | 2 | 100 | 256 | 1e-05 | 42 | completed |
| `dapo_m2_s100_resume50` | 2 | 100 | 256 | 1e-05 | 42 | completed |
| `dapo_m384_s100` | 384 | 100 | 256 | 1e-05 | 42 | completed |
| `dapo_m4_s100` | 4 | 100 | 256 | 1e-05 | 42 | completed |
| `dapo_m8_s100` | 8 | 100 | 256 | 1e-05 | 42 | completed |
| `deepmath_full_s100` | full pool (shuffled) | 100 | 256 | 1e-05 | 42 | completed |
| `deepmath_m1_s100` | 1 | 100 | 256 | 1e-05 | 42 | completed |
| `deepmath_m3840_s100` | 3840 | 100 | 256 | 1e-05 | 42 | completed |
| `deepmath_m48_s100` | 48 | 100 | 256 | 1e-05 | 42 | stopped at step 75 |
| `deepmath_m8_s100` | 8 | 100 | 256 | 1e-05 | 42 | completed |

## support_size / light_r1_7b_to_r1_distill_7b

student `r1_distill_qwen_7b`; teacher `light_r1_7b_ds`

| run | M | steps | batch | lr | seed | status |
|---|---|---|---|---|---|---|
| `dapo_m8_s50` | 8 | 50 | 256 | 1e-05 | 42 | completed |
| `dapo_m8_s50_resume20` | 8 | 50 | 256 | 1e-05 | 42 | completed |
| `deepmath_m8_s50` | 8 | 50 | 256 | 1e-05 | 42 | completed |
| `light_stage2_full_s50` | 3533 | 50 | 256 | 1e-05 | 42 | completed |

## support_size / math_grpo500_to_qwen3_4b

student `qwen3_4b`; teacher `math_grpo500_qwen3_4b`

| run | M | steps | batch | lr | seed | status |
|---|---|---|---|---|---|---|
| `deepmath_full_s165_b1024` | full pool (shuffled) | 165 | 1024 | 1e-05 | 42 | stopped at step 75 |
| `deepmath_m1_s15` | 1 | 15 | 256 | 1e-05 | 42 | completed |
| `deepmath_m3072_s15` | 3072 | 15 | 256 | 1e-05 | 42 | completed |
| `deepmath_m3840_s15` | 3840 | 15 | 256 | 1e-05 | 42 | completed |
| `deepmath_m384_s15` | 384 | 15 | 256 | 1e-05 | 42 | completed |
| `deepmath_m48_s15` | 48 | 15 | 256 | 1e-05 | 42 | completed |

## support_size / nemotron_1p5b_to_r1_distill_1p5b

student `r1_distill_qwen_1p5b`; teacher `nemotron_reasoning_1p5b_v1`

| run | M | steps | batch | lr | seed | status |
|---|---|---|---|---|---|---|
| `deepmath_m48_s100` | 48 | 100 | 256 | 1e-05 | 42 | stopped at step 75 |
| `eurus_code_m48_s100` | 48 | 100 | 256 | 1e-05 | 42 | completed |
| `mix3_m48_s100` | 48 | 100 | 256 | 1e-05 | 42 | stopped at step 75 |
| `textbook_science_m48_s100` | 48 | 100 | 256 | 1e-05 | 42 | stopped at step 75 |

## support_size / qwen3_30b_to_qwen3_1p7b

student `qwen3_1p7b`; teacher `qwen3_30b_a3b_instruct_2507`

| run | M | steps | batch | lr | seed | status |
|---|---|---|---|---|---|---|
| `dapo_full_s25_b1024` | full pool (shuffled) | 25 | 1024 | 1e-05 | 42 | completed |
| `dapo_m1_s15` | 1 | 15 | 256 | 1e-05 | 42 | completed |
| `dapo_m3840_s15` | 3840 | 15 | 256 | 1e-05 | 42 | completed |
| `dapo_m48_s15` | 48 | 15 | 256 | 1e-05 | 42 | completed |
| `dapo_m8_s15` | 8 | 15 | 256 | 1e-05 | 42 | completed |
| `deepmath_full_s50_b1024` | full pool (shuffled) | 50 | 1024 | 1e-05 | 42 | completed |
| `deepmath_m1_s15` | 1 | 15 | 256 | 1e-05 | 42 | completed |
| `deepmath_m3840_s15` | 3840 | 15 | 256 | 1e-05 | 42 | completed |
| `deepmath_m48_s15` | 48 | 15 | 256 | 1e-05 | 42 | completed |
| `deepmath_m8_s15` | 8 | 15 | 256 | 1e-05 | 42 | completed |

## support_size / qwen3_30b_to_qwen3_4b

student `qwen3_4b`; teacher `qwen3_30b_a3b_instruct_2507`

| run | M | steps | batch | lr | seed | status |
|---|---|---|---|---|---|---|
| `dapo_full_s25_b1024` | full pool (shuffled) | 25 | 1024 | 1e-05 | 42 | completed |
| `dapo_m1_s15` | 1 | 15 | 256 | 1e-05 | 42 | completed |
| `dapo_m3840_s15` | 3840 | 15 | 256 | 1e-05 | 42 | completed |
| `dapo_m48_s15` | 48 | 15 | 256 | 1e-05 | 42 | completed |
| `dapo_m8_s15` | 8 | 15 | 256 | 1e-05 | 42 | completed |
| `deepmath_full_s50_b1024` | full pool (shuffled) | 50 | 1024 | 1e-05 | 42 | completed |
| `deepmath_m1_s15` | 1 | 15 | 256 | 1e-05 | 42 | completed |
| `deepmath_m3840_s15` | 3840 | 15 | 256 | 1e-05 | 42 | completed |
| `deepmath_m48_s15` | 48 | 15 | 256 | 1e-05 | 42 | completed |
| `eurus_code_m48_s15` | 48 | 15 | 256 | 1e-05 | 42 | completed |
| `gsm8k_m1_s15` | 1 | 15 | 256 | 1e-05 | 42 | completed |
| `gsm8k_m48_s15` | 48 | 15 | 256 | 1e-05 | 42 | completed |
| `textbook_science_full_s50` | full pool (shuffled) | 50 | 256 | 1e-05 | 42 | completed |
| `textbook_science_m48_s15` | 48 | 15 | 256 | 1e-05 | 42 | completed |

## teacher_switch / math_or_code_grpo_to_qwen3_4b

student `qwen3_4b`; teacher `code_grpo300_qwen3_4b`, `math_grpo500_qwen3_4b`

| run | M | steps | batch | lr | seed | status |
|---|---|---|---|---|---|---|
| `code_teacher_deepmath_m48_sel42_train42_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `code_teacher_deepmath_m48_sel42_train43_s25` | 48 | 25 | 256 | 1e-05 | 43 | completed |
| `code_teacher_deepmath_m48_sel42_train44_s25` | 48 | 25 | 256 | 1e-05 | 44 | completed |
| `code_teacher_deepmath_m48_sel43_train42_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `code_teacher_deepmath_m48_sel44_train42_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `code_teacher_eurus_code_m48_sel42_train42_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `code_teacher_eurus_code_m48_sel42_train43_s25` | 48 | 25 | 256 | 1e-05 | 43 | completed |
| `code_teacher_eurus_code_m48_sel42_train44_s25` | 48 | 25 | 256 | 1e-05 | 44 | completed |
| `code_teacher_eurus_code_m48_sel43_train42_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `code_teacher_eurus_code_m48_sel44_train42_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `math_teacher_deepmath_m48_sel42_train42_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `math_teacher_deepmath_m48_sel42_train43_s25` | 48 | 25 | 256 | 1e-05 | 43 | completed |
| `math_teacher_deepmath_m48_sel42_train44_s25` | 48 | 25 | 256 | 1e-05 | 44 | completed |
| `math_teacher_deepmath_m48_sel43_train42_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `math_teacher_deepmath_m48_sel44_train42_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `math_teacher_eurus_code_m48_sel42_train42_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `math_teacher_eurus_code_m48_sel42_train43_s25` | 48 | 25 | 256 | 1e-05 | 43 | completed |
| `math_teacher_eurus_code_m48_sel42_train44_s25` | 48 | 25 | 256 | 1e-05 | 44 | completed |
| `math_teacher_eurus_code_m48_sel43_train42_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
| `math_teacher_eurus_code_m48_sel44_train42_s25` | 48 | 25 | 256 | 1e-05 | 42 | completed |
