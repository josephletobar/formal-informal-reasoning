# ABC Reusable Computation Publication Package

This folder contains the current publication-grade draft and its supporting figures, tables and reproducibility manifest.

## Core conclusion

The work reproduces an Anthropic-style attribution-graph and circuit-testing workflow on a small open model and synthetic ABC tasks. The model shows recurring answer-directed graph structure, positive in-sample path influence and causal source-position effects. The stronger held-out test does not show that discovery-selected recurrent features are necessary for addition relative to active controls, and cross-form activation transfer does not show operation-specific sufficiency.

The defensible scientific claim is therefore:

> Moderately simple reasoning problems can recruit recurring, answer-directed internal pathways, but recurrence and in-sample causal influence are insufficient to establish a reusable reasoning module.

## Contents

- `NMI_ARTICLE_DRAFT.md`: article-style manuscript.
- `make_publication_figures.py`: reproducible figure and summary-table generator.
- `figures/`: four figures in PNG and SVG.
- `tables/`: CSV metrics used by the figures and ledger.
- `REPRODUCIBILITY_MANIFEST.md`: raw-artifact map, commands and release blockers.
- `SUBMISSION_CHECKLIST.md`: journal-format and scientific-readiness checklist.
- `PREREGISTERED_REPLICATION_PROTOCOL.md`: fixed hypotheses, splits, controls and stopping rules for the next replication.
- `benchmark_v1/`: deterministic addition, subtraction and multiplication prompt benchmark plus manifest.
- `tables/artifact_hashes.json`: SHA256 hashes for the package and primary raw inputs.
- `clean_room_smoke.py`: package-only smoke test; it validates data counts and deserializes one graph without model inference.
- `audit_claims.py`: recomputes the manuscript's headline graph, path, held-out and transfer values from raw CSV files.
- `source_scripts/`: copies of the principal experiment drivers used to produce the mapped raw artifacts.
- `NMI_ARTICLE_SUBMISSION.docx`: Word initial-submission version of the article.
- `audit_submission_docx.py`: structural audit for the Word manuscript.
- `SUBMISSION_STATUS.md`: explicit local readiness and remaining external gates.
- `SUBMISSION_PAYLOAD.md`: exact manuscript, figure and supplementary-file upload list.
- `PRESUBMISSION_ENQUIRY_DRAFT.md`: optional editorial-scope enquiry draft.
- `run_independent_behavioral_screen.py`: bounded public-model behavioral screen run by CI; it is explicitly not a mechanistic replication.

The manuscript is a serious working draft, not evidence that a journal has accepted or will accept the paper. The remaining release blockers are explicitly listed so the project can move toward a credible submission rather than overclaiming from the current data.
