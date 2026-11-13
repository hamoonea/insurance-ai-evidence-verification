# Experimental protocol

Version 0.1, November 13, 2026. Development protocol, not a preregistration or a
completed study. The original protocol and CPU amendment preceded inference.
Execution status: the [first pilot results](../results/development-pilot-v1/README.md)
are now available; the evaluation rules below were not changed after generation.

## Research question

When a case lacks information needed to answer a scoped insurance-policy question,
does an explicit evidence-verification step reduce unsupported definitive answers
relative to a direct answer and a generic self-review?

The motivating idea is to check model output against external evidence, inspired
by [PLI-Analyzer](https://github.com/SuhasSrinivasan/plia). This implementation is
independent and does not copy that project's code. The healthcare-insurance
application and experimental comparison require their own evaluation; novelty
has not been established. Related work is listed in the repository README.

## Inputs and provisional labels

The current development set contains 12 fictional cases in four matched families,
with three variants per family. Every family has one provisional example of each
label: `supported`, `not_supported`, and `insufficient_information`. All cases and
labels are AI-drafted and await independent review. No patient records are used.

| Family | Policy | Scoped question |
| --- | --- | --- |
| F01 | CMS NCD 240.4, version 3 | Additional symptom/comorbidity condition in B.5.b |
| F02 | CMS NCD 240.4, version 3 | Prior physician order and supervision in B.4 |
| F03 | CMS NCD 210.14, version 2 | Current-smoking or recent-cessation criterion |
| F04 | CMS NCD 210.14, version 2 | Five beneficiary eligibility criteria |

The questions concern specified conditions. A negative label does not represent
a final coverage denial; other requirements can affect actual payment. Missing
information is different from documented absence. A known false AND condition can
resolve a negative answer despite another unknown condition; a true OR pathway can
resolve a positive answer. The evaluator implements these three-valued rules.

Policy snapshots contain normalized extracted sections A through D, citation
handles, source URLs, and version metadata. The saved extraction is not the
original HTML. The manifest records SHA-256 hashes. Independent review should
check both the saved source text and each case's interpretation.

## Comparison

All methods receive the same policy text and public case facts. Model input
excludes case IDs, family IDs, gold labels, and rationales.

1. **Direct:** Produce a label, explanation, citations, and any clarifications.
2. **Generic review:** Reconsider the exact direct answer for errors.
3. **Evidence verification:** Review that same direct answer, extract conditions,
   map them to policy and case evidence, and return an AND/OR expression.

The direct generation is shared by both review methods. Each complete comparison
requires three generations per case, or 36 for the current development set.
Case order and review order are seeded. Prompts, settings, raw outputs, token
counts, timing, and validation results are recorded. The planned local model is
Qwen3-4B-Instruct-2507 in a 4-bit MLX format; the first attempted run was blocked by
unavailable GPU access. There is no validated GPU result in this version.

The evidence gate verifies reference IDs, quoted-text presence, schema, and the
logic expression. A definitive criterion without quoted evidence becomes unknown.
Invented references or quotations make the output invalid, not an abstention.
Raw and gated outputs are retained separately. Matching text is not proof of
semantic entailment, and the gate cannot establish that every necessary criterion
was extracted. These limitations require human review.

## Measurements

The primary descriptive comparison is the rate of unsupported definitive answers
on cases whose provisional gold label is `insufficient_information`. Also report:

- Failure on incomplete cases including invalid or missing output.
- Answer rate on complete cases and error among definitive answers.
- Class recall, macro F1, confusion matrices, and invalid/missing counts.
- Evidence-verification performance before and after the deterministic gate.
- Human review of semantic evidence support, omitted criteria, and clarification
  usefulness.

Compare evidence verification with generic review using paired family bootstrap
resampling: variants and any repeats remain together. Four development families
are too few for broad conclusions. Do not treat variants or deterministic repeats
as independent observations. Report missing outputs rather than dropping them.

The fixture backend exists solely to test software. It abstains on every case;
its analysis intentionally contains no model-performance metrics. Neither
financial savings nor clinical outcomes are measured by this prototype.

## Next milestones

1. Independently review source fidelity, case scope, and provisional labels.
2. Run the development experiment, inspect failures, and document all revisions.
3. Expand toward 90 cases across 30 matched families, subject to review capacity.
4. Freeze a family-disjoint development/test split, prompts, metrics, and analysis
   choices before examining the held-out set. All variants stay in one split.
5. Run the frozen evaluation and publish failures and limitations alongside results.

The 90-case set and held-out evaluation are future work, not existing deliverables.
The present code and data can support feasibility work but do not establish
clinical reliability or superiority over another method.

## CPU pilot amendment — November 13, 2026

Before the first successful model generation, the execution plan was amended
because the local MLX runtime could not access a Metal GPU. The CPU pilot uses the
original `Qwen/Qwen3-4B-Instruct-2507` BF16 checkpoint, pinned to revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, through Transformers on standard
GitHub-hosted CPU runners. This is not the proposed MLX 4-bit conversion.

The [frozen run specification](../experiments/cpu-pilot-v1.json) preserves the
existing cases, gold labels, prompts, validator, and analysis definitions. Each
matched family runs on one worker; all three methods for a case share the same
initial answer. Greedy decoding, one repeat, seed 20261113, and a 2,000-token output
limit are fixed before generation. Failed or truncated outputs are retained.
The family shards must have identical settings, model fingerprints, and code/data
hashes before merging. This change addresses execution feasibility and does not
establish equivalence between the BF16 and MLX-quantized models.
