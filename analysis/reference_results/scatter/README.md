# Strict same-checkpoint parameter/function scatter

## Definition

Each point uses one Initial \(\theta_0\), same-architecture teacher \(T\), and
OPD endpoint \(D\):

\[
x=\cos(\theta_T-\theta_0,\theta_D-\theta_0),\qquad
y=\cos(v_T,v_D).
\]

Both axes read the same published HF teacher and endpoint weight files. \(x\)
is an exact all-parameter cosine after promoting the stored weights to FP32.
\(y\) reloads those same values in FP32 and computes the centered
frozen-top-K Fisher delta-log-probability cosine on Initial-generated states.

Cross-architecture 30B→4B pairs are excluded because a raw parameter cosine is
undefined.

## Dataset

| Teacher family | Points | Parameter range | Functional range | Spearman |
|---|---:|---:|---:|---:|
| Qwen3-4B / Math GRPO-500 | 7 | 0.010–0.093 | 0.755–0.852 | −0.357 |
| Qwen3-4B / Code GRPO-300 | 11 | 0.018–0.071 | 0.371–0.933 | 0.927 |
| DeepSeek-1.5B / JustRL | 9 | 0.044–0.091 | 0.883–0.974 | 0.967 |
| DeepSeek-1.5B / Nemotron | 4 | 0.134–0.181 | 0.923–0.977 | 1.000 |
| Qwen3-4B-Instruct / Mix teacher | 7 | 0.016–0.057 | 0.060–0.674 | 0.821 |
| DeepSeek-1.5B / DeepScaler | 2 | 0.131–0.134 | 0.960–0.963 | — |
| DeepSeek-7B / Light-R1-7B-DS | 3 | 0.076–0.139 | 0.859–0.895 | 0.500 |

Total: 43 points. Parameter cosine ranges 0.0096–0.1808; functional cosine
ranges 0.0598–0.9775. Medians are 0.069 and 0.859 respectively, with a median
vertical gap of 0.780. Every point has \(y>x\), and 32/43 have \(x<0.2\) and
\(y>0.7\).

Overall Pearson is 0.585 and Spearman 0.664. These are descriptive only: rows
share teachers, histories and sometimes a training trajectory, so a naive
43-point regression p-value is invalid.

## Representative points

| Pair | Parameter cosine | Functional cosine |
|---|---:|---:|
| Math GRPO-500 → OPD-s50 | 0.0917 | 0.8501 |
| Math GRPO-500 → M1-Code | **0.0096** | **0.8523** |
| Code GRPO-300 → A′-s100 | 0.0178 | 0.3713 |
| Code GRPO-300 → Recovery | **0.0285** | **0.8661** |
| Code GRPO-300 → FULL | 0.0692 | 0.9327 |
| JustRL → DAPO-M384 | 0.0906 | 0.9741 |
| Nemotron → Mix48 | 0.1808 | 0.9775 |
| DeepScaler → DAPO-M8 | 0.1345 | 0.9631 |
| Light-R1-7B → stage2 | 0.1386 | 0.8953 |
| Mix teacher → Math48 | 0.0553 | 0.6736 |
| Mix teacher → IF48 | 0.0163 | 0.0598 |

## Interpretation

The robust cross-family statement is:

> High teacher-aligned functional change does not require a parameter update
> aligned with the teacher's raw task vector.

The scatter does not support the converse “low parameter cosine implies high
functional cosine.” Mix-teacher IF48 is low on both axes, and A′ endpoints have
low parameter alignment with only moderate or low functional alignment.

Parameter alignment can still rank endpoints within some families: Code,
JustRL, Nemotron and Mix teacher are positively ordered. Math GRPO-500 is the
counterexample—its parameter cosine does not rank functional reproduction.
Thus raw parameter direction is neither necessary nor a family-invariant
predictor.

The Code trajectory is especially discriminating. A′-s100 and Recovery have
similarly tiny parameter cosines (0.0178 and 0.0285), while their functional
cosines are 0.371 and 0.866. A small horizontal movement accompanies a large
functional reorganization.

## Precision audit

The main plot intentionally retains the original strict same-HF Math pair:
\((0.0917, 0.8501)\). The native FSDP FP32 optimizer-master cosine for the same
training endpoint is 0.1619. That native number answers a different question
because the functional forward uses the merged-HF values. The earlier v1 plot
is kept as a supplementary optimizer-master precision audit and is explicitly
marked superseded for the main same-checkpoint claim.

## New Light-R1 7B probe

The 7B points required a new cohort because its vocabulary differs from the
1.5B family. The cohort freezes 32 Initial-generated trajectories each from
Math, GPQA, HumanEval+ and LiveCodeBench-v6 (128 total, 130,414 positions).
Initial, teacher and all three endpoints were scored in FP32 on the same
top-K=128 support:

- DAPO-M8: (0.0791, 0.8590)
- DeepMath-M8: (0.0756, 0.8612)
- Light-stage2: (0.1386, 0.8953)

## Limitations

1. Functional geometry is conditional on Initial-visited states and the frozen
   Initial top-K support.
2. Families use separate Initial-generated cohorts; the definition is shared,
   not the exact token sequences.
3. Multiple rows within a family are not independent.
4. Same-HF consistency avoids cross-axis checkpoint mismatch, but does not
   preserve the optimizer's FP32 master weights.
5. A high functional cosine does not imply identical on-policy occupancy,
   benchmark score, or internal representation.

## Artifacts

- Figure: `parameter_vs_functional_cosine.png` and `.pdf`
- Audited points: `scatter_data.csv`, `scatter_data.json`
- Builder: `scripts/analysis/build_parameter_function_scatter_expanded.py`
- Same-HF parameter tool:
  `scripts/analysis/hf_task_vector_cosine.py`
- Light 7B function geometry:
  `results/analysis/light_r1_7b_functional_geometry_v1/`
- DeepScaler function geometry:
  `results/analysis/deepscaler_1p5b_functional_geometry_v1/`
- Extended Mix function geometry:
  `results/analysis/source_domain_functional_recovery_v1/mixrl/initial_probe/collinearity_extended.json`
