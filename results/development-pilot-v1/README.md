# First model experiment: mixed results and three structural failures

**November 13, 2026 · 12 development cases · 36 genuine model generations**

Generic review answered 9/12 cases correctly, compared with 8/12 for evidence
verification and 6/12 for direct answers. Evidence verification made fewer
definitive claims on incomplete cases, but three of its outputs failed validation.
It did not improve the overall rate of resolving incomplete cases correctly over
generic review. This pilot therefore does **not** establish evidence verification
as the better method.

These are exploratory results against AI-drafted, provisional labels. No coverage
expert has independently reviewed the cases or outputs, and there is no held-out
test set. The questions address named policy conditions, not final claim decisions.

## Results

There are four cases with missing decisive information and eight complete cases.
Invalid outputs count as incorrect. They are never counted as successful abstention.

| Measure | Direct | Generic review | Evidence verification, after gate |
| --- | ---: | ---: | ---: |
| Correct labels, all 12 cases | 6/12 (50.0%) | 9/12 (75.0%) | 8/12 (66.7%) |
| Unsupported definitive answers, 4 incomplete cases | 4/4 | 2/4 | 1/4 |
| Incomplete-case failures, including invalids | 4/4 | 2/4 | 2/4 |
| Definitive answer rate, 8 complete cases | 8/8 | 8/8 | 6/8 |
| Invalid outputs | 0/12 | 0/12 | 3/12 |
| Incorrect among definitive answers | 6/12 | 3/10 | 1/7 |
| Macro F1 | 0.389 | 0.741 | 0.746 |

The evidence method's 1/4 unsupported-definitive rate must be read alongside its
1/4 invalid rate on incomplete cases: both methods correctly returned insufficient
information on only 2/4 incomplete cases. Evidence verification lost two additional
complete cases to invalid outputs. Its slightly higher macro F1 reflects class
balance and does not override those failures or its lower total accuracy.

The exploratory family bootstrap estimates an evidence-minus-generic difference
of −25 percentage points in unsupported definitiveness, with a percentile interval
of [−75, 0] points. Only four families were resampled. This interval is descriptive;
it does not establish a reliable general improvement and does not measure the
invalid-inclusive outcome.

## Complete case ledger

`S` = supported; `N` = not supported; `U` = insufficient information; `X` = invalid.
Labels refer only to each question's explicitly named requirement.

| Case | Family | Provisional gold | Direct | Generic | Evidence after gate |
| --- | --- | --- | --- | --- | --- |
| D001 | F03: smoking recency | S | N | S | S |
| D002 | F04: LDCT beneficiary criteria | S | N | N | S |
| D003 | F03: smoking recency | U | N | U | U |
| D004 | F01: additional symptom/comorbidity | N | N | N | X |
| D005 | F02: order and supervision | N | N | N | X |
| D006 | F03: smoking recency | N | N | N | N |
| D007 | F02: order and supervision | S | S | S | S |
| D008 | F04: LDCT beneficiary criteria | N | N | N | N |
| D009 | F02: order and supervision | U | S | S | N |
| D010 | F04: LDCT beneficiary criteria | U | N | S | U |
| D011 | F01: additional symptom/comorbidity | S | S | S | S |
| D012 | F01: additional symptom/comorbidity | U | S | U | X |

## What failed, and what the gate changed

This is an AI-assisted inspection of saved outputs, not independent annotation.

1. **A valid quotation did not establish the claimed condition (D009).** The only
   relevant fact confirms physician supervision; prior physician ordering is
   unknown. Direct and generic answers nevertheless returned `supported`.
   The verifier marked the combined order-and-supervision condition `false`, using
   the supervision quote. Its raw label was `supported`; the gate changed it to
   `not_supported`. Both labels are wrong against the provisional `U` key.
   Exact quotation matching and deterministic logic cannot repair incorrect
   interpretation of the evidence.
2. **Three outputs used malformed logic (D004, D005, D012).** Each placed `all`
   and `any` in the same node, violating the one-operator grammar. D005 also had
   an empty `any` array; D004 introduced an out-of-scope sleep-duration condition
   and omitted it from the logic. All three were rejected with
   `Malformed or overly deep logic tree`. Their raw label fields happened to
   match the answer keys, but they remain invalid in all primary results. No
   repair, retry, or post-hoc acceptance was applied.
3. **Review invented a missing fact (D010).** Generic review asserted that the
   beneficiary would receive an LDCT order, although no order fact was supplied.
   Evidence verification instead marked that criterion unknown; the gate changed
   its raw `not_supported` label to `insufficient_information`, correctly.
4. **The label contradicted the explanation (D001, D002).** D001's direct answer
   kept a negative label even though its prose identified the positive conclusion.
   Both reviews fixed it. D002's direct and generic outputs also retained a
   negative label while their prose said the five criteria were met. The evidence
   output represented all five as true, so the gate corrected its negative label.
5. **A missing comorbidity was treated as satisfied (D012).** The direct answer
   acknowledged that symptoms/comorbidities were missing but still returned
   `supported`. Generic review correctly returned `insufficient_information`.
   Evidence review recognized the missing information but failed the logic grammar.

Across the nine structurally valid evidence outputs, the gate changed three labels:
D002 and D010 became correct; D009 changed from one wrong definitive answer to
another. Keeping the three invalid outputs as failures, evidence accuracy rose
from 6/12 for pre-gate labels to 8/12 after gating. This is a within-output check,
not a separately generated ablation.

## Model, execution and cost observations

The frozen model was `Qwen/Qwen3-4B-Instruct-2507`, original BF16 weights, revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. Four GitHub CPU workers each processed
one whole family. Settings were temperature 0, one repeat, seed 20261113,
2,000 maximum new tokens, PyTorch 2.9.1+cpu, Transformers 4.57.3, and four CPU
threads. All 36 generations stopped normally; none hit the token limit.

The initial MLX attempt could not access a Metal GPU. The switch to original BF16
CPU weights was committed before inference in the
[CPU amendment](../../experiments/cpu-pilot-v1.json). These results are not a
measurement of the original 4-bit MLX configuration. Prompts, cases, provisional
keys and scoring code were frozen before this run.

| Generation stage, 12 calls each | Output tokens | Mean seconds per call | Sum of measured call seconds |
| --- | ---: | ---: | ---: |
| Direct | 2,028 | 132.3 | 1,587.4 |
| Generic review only | 1,976 | 138.8 | 1,666.1 |
| Evidence review only | 5,841 | 352.5 | 4,230.4 |

A standalone generic pipeline also needs the initial answer: 4,004 output tokens
across 12 cases, averaging 271.1 seconds per case including that call. The analogous
evidence pipeline used 7,869 output tokens across 12 cases and averaged 484.8 seconds
per case. In this experiment
the initial answer was shared, so actual total generation was 9,845 output tokens
across 36 calls. Timings exclude setup/download and are affected by different CPU
workers. Sums across parallel workers are not wall-clock duration. No API prices,
financial savings, clinical benefit, or production latency are estimated.

## Audit and reproduction

- [Frozen inference source](https://github.com/hamoonea/insurance-ai-evidence-verification/tree/1b91d554214d07f7a88656c155a477b4e412c8c4)
- [Successful inference run](https://github.com/hamoonea/insurance-ai-evidence-verification/actions/runs/43489781164)
- [Successful frozen-source merge and analysis](https://github.com/hamoonea/insurance-ai-evidence-verification/actions/runs/43496819358)
- [Machine-readable metrics](analysis.json), [case ledger](case_results.csv),
  [raw responses](responses.jsonl), and [model/configuration provenance](provenance.json)

All 36 responses, including failures, are retained. Reconstructed prompt hashes,
frozen input/code hashes and output-validation decisions were verified. Local
rescoring exactly matched all case-level rows and matched summary numbers within
1e-12 across Python versions. `analysis.json` preserves the original runner's
values. Raw responses were recovered from the analysis job's structured log
export; [audit details](audit.json) identify that source. GitHub Actions artifacts
have limited retention; the published result files are the durable record.

Replay scoring and generate the actual-output review packet without a model or API:

```sh
python3 scripts/replay_pilot_results.py --out outputs/replayed-development-pilot
```

The replay verifies all frozen file hashes and refuses changed sources. If future
code changes, use the frozen inference checkout with the replay script and result
folder from this results publication. Replay is a check of existing responses;
it does not rerun inference. New inference can vary with runtime and hardware.

## Next experiment

First obtain independent review of the 12 cases and policy interpretations.
For the next development version, test an explicit logic grammar example or
schema-constrained generation, require atomic order and supervision conditions,
and audit whether evidence actually supports each condition. Compare those
changes against both existing baselines. Keep this pilot immutable, register any
repair/retry policy in advance, and include extra calls in cost accounting.

Expand toward the planned 90 cases with family-separated splits and expert-reviewed
labels. Freeze the revised prompts and test set before running a held-out
evaluation. Do not use improved scores on these already-inspected 12 cases as
evidence of generalization.
