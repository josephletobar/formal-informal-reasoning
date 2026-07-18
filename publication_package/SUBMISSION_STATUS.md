# Submission Status

**As of:** 2026-07-17  
**Target:** Nature Machine Intelligence, Article

## Ready locally

- Audited manuscript in Markdown and Word format.
- 140-word abstract and 1,584-word main-text body in the audited draft.
- Four display figures in PNG and SVG formats.
- Cover-letter draft and author-metadata template.
- Raw tables, benchmark generator and experiment-driver copies. Serialized graph objects remain in the local archive but are not included in this lightweight public release.
- SHA256 manifest covering 27 package, data and driver inputs.
- Claim audit, clean-room package smoke test and package verifier all pass.
- A bounded public-model behavioral screen is now configured in CI; its artifact is kept separate from the mechanistic claims.
- Submission archive: `publication_package_submission_checkpoint.zip`.

## Not yet complete

1. The author must supply names, affiliations, corresponding-author contact details and declarations. The public repository is now available at the URL recorded in `PUBLIC_RELEASE.md`.
2. Exact model and transcoder snapshot hashes still need to be recorded from the runtime that performed the experiments.
3. Visual Word/PDF render QA requires a machine with LibreOffice or another Word renderer; the current machine has no usable `soffice` renderer.
4. The preregistered independent replication has not run. The RunPod endpoint is currently stopped, and the local GPU is occupied by unrelated processes.
5. The corresponding author must log into Nature's submission system and upload the manuscript and associated files. Acceptance and publication can only be decided by the journal.

## Scientific submission position

The current evidence supports recurring answer-directed pathways and in-sample causal influence, but not a discrete reusable reasoning module. The held-out recurrent-feature necessity test is null. The manuscript therefore makes a bounded mechanistic claim and explicitly identifies the independent replication as the next evidentiary gate.
