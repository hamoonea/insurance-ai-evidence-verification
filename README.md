# Insurance AI Evidence Verification

**v0.1 · Research prototype · No live model results yet**

This research prototype tests whether checking policy conditions against supplied
facts can reduce definitive answers when required information is missing.
The [experimental protocol](docs/protocol.md) describes the comparison and its
limits. This is an independent, AI-assisted research project by Iftekhar Ahmad.

**Current status:** 12 fictional development cases, two versioned CMS policy
extracts, three comparison methods, a local MLX model adapter, reproducible run
records, automated label analysis, and a review packet exporter are implemented.
The cases and answer keys are AI-drafted and await independent review. There are
no held-out test cases yet. Software fixtures are never reported as model results.

## Quick start

The validator, tests, fixture run, and analyzer use Python 3.10+ standard library
only. Clone the repository and run from its directory:

```sh
git clone https://github.com/hamoonea/insurance-ai-evidence-verification.git
cd insurance-ai-evidence-verification
python3 -m insurance_eval validate
python3 -m unittest discover -s tests -v
python3 -m insurance_eval run --backend fixture --out outputs/software-smoke
python3 -m insurance_eval analyze --out outputs/software-smoke
python3 -m insurance_eval review --out outputs/software-smoke
```

The fixture deliberately abstains on every case. Its only purpose is to test file
flow and output validation. The analyzer suppresses model performance statistics
for fixture runs.

## Run a local model

Use an Apple Silicon environment with GPU access, `mlx-lm` installed, and an
existing compatible model directory. The runner does not download model weights,
use an API, or send the cases outside the machine.

In a separate Python environment, install the optional dependencies with
`python3 -m pip install -r requirements-mlx.txt`. These are the versions detected
in the initial local environment; GPU execution has not been validated here.

```sh
python3 -m insurance_eval run --backend mlx \
  --model-path /absolute/path/to/local/model \
  --out outputs/development-model-v1 --repeats 1
python3 -m insurance_eval analyze --out outputs/development-model-v1
python3 -m insurance_eval review --out outputs/development-model-v1
```

The initial runtime has `mlx==0.32.4` and
`mlx-lm==0.32.2`. Its model manifest identifies
`mlx-community/Qwen3-4B-Instruct-2507-4bit`, revision
`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`. This model and environment are not
bundled here. Importing MLX in the initial execution session failed because no Metal device was
available. No live model findings were produced by that failed attempt.

Choose a new output directory after changing code, inputs, generation settings,
or model. Identical requests can resume, while incompatible configurations are
rejected. At temperature zero, repeated generations may be identical; repetitions
are not independent samples. Start with one repeat for development.

## Experimental comparison

All methods receive the same complete substantive sections A through D from the
selected policy version, plus the same question and fictional case facts.

| Method | Calls per case | Behavior |
| --- | --- | --- |
| Direct | One | Produces a scoped label, explanation, citations and clarifications |
| Generic review | Initial call plus review | Revises that exact initial answer |
| Evidence verification | Same initial call plus review | Extracts conditions, maps case evidence, and evaluates true/false/unknown logic |

The initial call is shared. Twelve cases therefore need 36 generation calls for
one full comparison (12 initial calls, 12 generic reviews, 12 evidence reviews).
Policy and case text are treated as evidence, not instructions. IDs, family
metadata, gold labels and rationales are excluded from model inputs.

The evidence gate checks that references exist and quotations occur in the
specified fact. It evaluates AND/OR expressions without executing model code.
Definitive conditions with no quoted evidence become unknown; invented references
or quotations make the output invalid. This is **not a semantic correctness
guarantee**. The model can cite real text incorrectly or omit a required criterion.
Human review must check those failures. Raw answers, gate changes and final labels
are all preserved.

## Data and scope

The 12 cases comprise four families with three variants each: supported,
not supported, and insufficient information. Family variants differ in one
decisive fact. The two CPAP families concern an additional symptom/comorbidity
condition and the order/supervision requirement. The two LDCT families concern
smoking recency and the five beneficiary criteria.

Every question identifies its scope. A negative criterion result is not a claim
denial; actual Medicare payment can depend on additional requirements. These are
fictional cases and not patient records.

* `data/cases/development.jsonl`: public model inputs and split metadata.
* `data/gold/development.jsonl`: provisional labels, evidence and review status.
* `data/policies/`: normalized official policy text with citation segment IDs.
* `data/raw/`: retained source extraction before Unicode normalization.
* `data/manifest.json`: SHA-256 input hashes and provenance.
* `insurance_eval/`: prompts, inference, evidence gate and analysis.
* `tests/`: meaningful failure, leakage, metric and integration checks.
* `docs/reviewer-guide.md`: annotation and review instructions.

Generated run records, environments, model weights, and machine-specific logs are
excluded from version control. The quick-start commands regenerate the software
check and review packet locally.

Sources were retrieved through web extraction on November 13, 2026. Direct shell
downloads could not resolve the CMS host. These files are labeled as normalized
extractions, not original HTML. Source line IDs are local citation handles, not
official CMS clause numbers. Extracts include policy exceptions, but not separately
linked documents or generic website help. Before expansion, independently compare
the saved text and labels with the linked source versions.

## Analysis

Report unsupported definitive answers on incomplete cases together with answer
rate on complete cases, selective error, per-class recall, macro F1, confusion
matrices, and invalid/missing output counts. Errors are not silently treated as
abstentions. Evidence-verified raw labels are scored separately from gated labels.

Family bootstrap comparisons keep variants and repeats together. Four development
families cannot establish broad performance. Evidence entailment, omitted criteria
and clarification usefulness remain human-scored. No automated LLM judge is used.
Latency and token counts are retained per generation; reviews share initial-call
cost. The prototype does not estimate financial savings or patient outcomes.

## Next research milestone

Review these 12 cases with a coverage-review expert, execute one development run,
inspect its failures, and then expand to the planned 90 cases. Freeze family splits
and prompts before evaluating the future held-out set. The present package is a
feasibility prototype, not a completed study or a validated coverage system.

## Sources and prior work

* [CMS NCD 240.4 version 3](https://www.cms.gov/medicare-coverage-database/view/ncd.aspx?ncdid=226&ncdver=3)
* [CMS NCD 210.14 version 2](https://www.cms.gov/medicare-coverage-database/view/ncd.aspx?ncdid=364&ncdver=2)
* [PLI-Analyzer](https://github.com/SuhasSrinivasan/plia): inspiration for evaluating model output against external evidence; no code copied.
* [Health insurance RAG with an evaluator agent](https://www.nature.com/articles/s41598-025-31038-6)
* [Reimbursement reasoning and incomplete cases](https://www.jmir.org/2026/1/e95877)
* [InsClaimBench](https://arxiv.org/abs/2610.09671)
* [MLX LM documentation](https://github.com/ml-explore/mlx-lm)

The work is positioned as a focused replication and extension. Publication-level
novelty and clinical usefulness are not established.

## Development and attribution

AI assistance was used to draft the implementation, documentation, synthetic
cases, and provisional answer keys. Passing software tests does not validate
those annotations or establish research findings. See the
[implementation status](docs/implementation-status.md) and
[reviewer guide](docs/reviewer-guide.md) for remaining work.

The source policy excerpts retain their CMS attribution and version information.
This project is not affiliated with or endorsed by CMS, an insurer, or a university.
It is a research tool, not a system for deciding an individual's coverage or care.
