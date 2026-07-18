# Public Release

The audited research package is publicly available in the repository's canonical location:

- Repository: https://github.com/josephletobar/formal-informal-reasoning
- Release tag: `v0.1.9-abc-publication`
- The tag is the authoritative immutable release reference.

The current main branch includes later metadata and workflow commits. The release tag points to the corrected dependency-pinned behavioral-screen workflow.

The release contains the article draft, Word submission file, figures, benchmark, repository-relative derived data, source drivers and verification scripts. It does not contain model weights, gated credentials or the large serialized graph archive retained locally.

From a fresh checkout, the lightweight checks are:

```bash
python publication_package/audit_claims.py
python publication_package/clean_room_smoke.py
python publication_package/verify_publication_package.py
python publication_package/make_publication_figures.py
```

The release is a public research artifact, not a journal publication or acceptance notice.
