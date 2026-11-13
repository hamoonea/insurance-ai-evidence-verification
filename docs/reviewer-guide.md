# Reviewing the development benchmark

The answer key is AI-drafted and has not been independently validated. A reviewer
with coverage-review expertise should first verify the policy version and each
case's task scope. The questions ask about specified criteria, not actual payment.

For each case:

1. Read the applicable policy sections, including alternative pathways.
2. Identify which conditions are required for the exact question.
3. Map every supplied fact to those conditions without filling gaps.
4. Assign supported, not_supported, or insufficient_information.
5. Record the source clauses, rationale, and any required missing fields.
6. Flag ambiguities and revise the case before freezing an evaluation split.

A documented failure of a necessary AND condition can settle a negative answer
even if another condition is unknown. One satisfied OR pathway can settle a
positive answer. Silence alone establishes neither presence nor absence.

For model outputs, generate the review packet with `python3 -m insurance_eval
review --out outputs/RUN`. The packet conceals method names, but structure may
reveal a method. Keep `private_method_key.json` separate from reviewers. For a
fully independent label review, provide public cases and source policies before
showing the provisional gold key.

The CSV leaves all review columns blank. Suggested values are yes/no/uncertain
for label correctness, policy support, case support and condition completeness;
use not_applicable for clarification fields when no fact is missing. Document any
accepted alternative phrasing of a missing field. Score an irrelevant request for
more information as a clarification error. Report actual review coverage and
adjudication of disagreements. Do not replace unknowns with assumed negatives.

Do not interpret exact quote matching as evidence of semantic support. A quotation
can be present yet contradicted by surrounding text or used with the wrong polarity.
