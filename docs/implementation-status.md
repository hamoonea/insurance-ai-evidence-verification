# Implementation status — November 13, 2026

This is a development prototype. It contains 12 fictional, AI-drafted cases
from four matched families across two CMS policy versions. All provisional
answer keys require independent review. No held-out test set exists yet.

## Completed checks

- All 26 unit and integration tests passed, including family-shard integrity checks.
- Frozen dataset and source hashes passed validation.
- All 36 outputs from the full software-fixture run passed output validation.
- The fixture analyzer correctly suppressed model-performance statistics.
- Method-concealed review exports and a blank annotation sheet were generated.
- All 36 genuine CPU model responses were saved, including three invalid outputs.
- Frozen-source merging, prompt-hash reconstruction, revalidation, and scoring replay passed.

Fixture outputs are intentionally synthetic software checks, not model findings.
The review packet generated under `outputs/software-smoke/review` demonstrates
the export format only; it should not be scored as a live experiment. Generated
outputs are excluded from the public repository and can be recreated with the
README commands.

## Live experiment

A local MLX model run was attempted using the existing Qwen3-4B-Instruct-2507
4-bit model. MLX initialization failed because this execution session has no
accessible Metal GPU. The failed attempt produced a configuration and blocked
status record, but no model answers or performance measurements.

The execution plan was amended before inference to use the original BF16
`Qwen/Qwen3-4B-Instruct-2507` checkpoint on four GitHub CPU workers, with one matched
family per worker. All four jobs and the frozen-source merge/analysis completed
successfully. These are actual model generations, not fixture outputs or a
measurement of the unavailable MLX environment.

The [published pilot](../results/development-pilot-v1/README.md) contains all 36 raw
responses, scores, model fingerprints and failure analysis. Direct answers scored
6/12, generic review 9/12, and evidence verification 8/12. Evidence verification
produced three malformed logic expressions. Its lower unsupported-definitive rate
did not improve the invalid-inclusive incomplete-case outcome over generic review.

All 36 prompt hashes and output-validation decisions reproduced. All 16 frozen
input/code hashes matched. Case-level scores reproduced exactly, with summary
metrics matching within 1e-12 across Python versions. The published metrics retain
the original runner's values. A replay script regenerates scoring and the actual
response review packet without model dependencies.

Independent review of policy fidelity, provisional labels, semantic evidence
support, omitted conditions, and clarification usefulness remains pending.

## Research claims

The small development comparison is reported descriptively. No general accuracy
improvement, financial savings, clinical benefit, or publication-level novelty is
claimed. Semantic evidence correctness and omitted criteria remain matters for
human review; valid quotation matching alone does not establish entailment.
