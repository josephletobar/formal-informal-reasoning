# Nature Machine Intelligence Submission Checklist

## Format

- [x] Article structure: Introduction, Results, Discussion, Methods.
- [x] Main text is below the journal's 3,500-word Article guideline.
- [x] Abstract is below 150 words and contains no references.
- [x] Four display figures are available in PNG and SVG formats.
- [x] Word initial-submission manuscript generated and structurally audited.
- [ ] Visual DOCX/PDF render QA on a machine with LibreOffice or another Word renderer.
- [x] The draft stays below the six-display-item guideline.
- [ ] Final reference formatting and author metadata.
- [ ] Final figure legends and supplementary-information cross-references.

## Scientific completeness

- [x] Behavioral, representation, graph, path, attention-source and causal results are separated.
- [x] Discovery/confirmation split is documented.
- [x] Active same-position controls are documented.
- [x] Failed runs and implementation fixes are retained rather than silently discarded.
- [x] Claims are limited where the held-out necessity test is null.
- [ ] Independent preregistered replication with larger discovery and confirmation sets.
- [ ] At least one additional model with a validated sparse-transcoder bundle.
- [x] Local package smoke test from the package boundary; full fresh-environment inference rerun remains open.

## Data and code

- [x] Raw CSVs are retained.
- [x] Serialized attribution graphs are retained for the primary panel.
- [x] Figure-generation script is included.
- [x] Reproducibility manifest identifies primary artifacts and commands.
- [x] Record the observed dependency environment in `environment-lock.txt`; convert to a tested install lock before release.
- [x] Add hashes for benchmark, report-input and graph-analysis artifacts; model/transcoder snapshot hashes remain open.
- [ ] Remove machine-specific absolute paths from public manifests.
- [x] Include deterministic benchmark generator and prompt manifest in the package.
- [ ] Add a license and public repository release.

## Editorial positioning

The journal describes an Article as a substantial novel research study using several techniques and permits up to 3,500 words of main text, a 150-word abstract and six display items. The present study fits the technical scope of machine learning and AI, but the current evidence is best positioned as a rigorous benchmark and mechanistic-audit contribution, not as proof that discrete reasoning modules exist.

Official pages:

- https://www.nature.com/natmachintell/aims
- https://www.nature.com/natmachintell/content
- https://www.nature.com/natmachintell/submission-guidelines
