# Public Release

The audited research package is publicly available in the repository's canonical location:

- Repository: https://github.com/josephletobar/formal-informal-reasoning
- Release tag: `v0.1.0-abc-publication`
- Commit: `def6f111226666881f954743f01c4e63f040bbb2`

The current main branch includes later metadata and workflow commits; the next stable tag will be `v0.1.5-abc-publication` after the workflow commit is pushed.

The release contains the article draft, Word submission file, figures, benchmark, repository-relative derived data, source drivers and verification scripts. It does not contain model weights, gated credentials or the large serialized graph archive retained locally.

From a fresh checkout, the lightweight checks are:

```bash
python publication_package/audit_claims.py
python publication_package/clean_room_smoke.py
python publication_package/verify_publication_package.py
python publication_package/make_publication_figures.py
```

The release is a public research artifact, not a journal publication or acceptance notice.
