# Step 2.6R-3B2 Independent Review and Gold Freeze

This isolated, standard-library-only package validates and freezes two
independent score-blind direct-relevance reviews, audits descriptive agreement,
creates an independently blinded disagreement-only adjudication packet, and
freezes the final relevance gold plus the preregistered 24/30 coverage gate.

It never loads or scores a selector, model, Frozen G1, checkpoint, prediction,
or veracity output. It does not train, construct an optimizer, access Formal
Validation/Test, or open the six sealed historical challenge contents.

## Phase 3B2-A

`run_freeze_reviews.py` performs these operations in order:

1. verify the public 3B1-R2 source artifacts and SHA sidecars;
2. verify the 30-case/289-unit source and public packet blindness contracts;
3. validate Reviewer A's exact immutable fields, labels, confidence, note, and
   provenance;
4. independently validate Reviewer B under the same contract;
5. freeze both exact reviewer returns and hashes;
6. only then verify/open the private mapping, align the same 289 underlying
   units, compute descriptive four-class and binary agreement plus Cohen's
   kappa, and freeze agreed/disagreement rows;
7. if disagreements exist, generate a new deterministic selector-blind packet
   using salt `step2.6r-3b2-adjudication-v1`.

Agreement has no acceptance threshold. Confidence and notes never alter labels.
Agreement rows freeze directly; disagreements remain `NEEDS_ADJUDICATION`.

## Phase 3B2-B

`run_freeze_adjudication.py` is a separate command. It revalidates and
recomputes the frozen A/B agreement chain. With disagreements, it validates the
exact adjudication template and provenance, and permits adjudication labels for
those rows only. With zero disagreements, no adjudication inputs are accepted.

Final labels use exactly:

- agreement: the shared A/B label, source `REVIEWER_AGREEMENT`;
- disagreement: the frozen adjudicator label, source
  `INDEPENDENT_ADJUDICATION`.

The final binary target is `DIRECT -> 1`; `RELATED`, `IRRELEVANT`, and
`UNREADABLE` map to `0`. Every one of the 289 units and all 30 cases remain in
the gold. A case is evaluable only when it has at least one `DIRECT` unit. The
coverage gate passes at 24 or more evaluable cases; failure freezes the gold
with `INDEPENDENT_AUDIT_RELEVANCE_COVERAGE_INSUFFICIENT`, performs no
resampling, and leaves Step 3B3 blocked.

This stage does not compute MRR, NDCG, Recall, rankings, or selector scores.
