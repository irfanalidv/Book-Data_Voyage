# talentlens-core — PyPI Publish Guide

## Package validation: 10/10 checks passed ✅

## Step 1 — Install build tools

```bash
pip install build twine
```

## Step 2 — Build the distribution

```bash
cd book/ch22
python -m build
# Creates: dist/talentlens_core-X.Y.Z-py3-none-any.whl
#          dist/talentlens_core-X.Y.Z.tar.gz
```

## Step 3 — Test on TestPyPI first

```bash
# Register at test.pypi.org (free)
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ talentlens-core
python -c "import talentlens_core; print(talentlens_core.__version__)"
```

## Step 4 — Publish to real PyPI

```bash
# Register at pypi.org (free)
# Generate an API token under Account Settings
twine upload dist/*
# Enter: __token__ as username, your token as password
```

## Step 5 — Verify

```bash
pip install talentlens-core
python -c "from talentlens_core import classify_role; print(classify_role('ML Engineer'))"
```

## Step 6 — Set up automated releases (GitHub Actions)

Add `PYPI_API_TOKEN` to GitHub repo Secrets.
The workflow in `book/ch22/.github/workflows/publish.yml` triggers on
`git tag v*` and publishes automatically.

```bash
git tag v1.0.1
git push origin v1.0.1
# GitHub Actions runs tests → build → publish
```

## Versioning convention

- `v1.0.0` — first public release
- `v1.0.1` — bug fix
- `v1.1.0` — new feature (backwards compatible)
- `v2.0.0` — breaking change

## Your package URL

Once published: https://pypi.org/project/talentlens-core/
