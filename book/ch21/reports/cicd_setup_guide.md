# TalentLens CI/CD Setup Guide

## Current status
| File | Status |
|------|--------|
| `.github/workflows/ci.yml` | Ready |
| `Makefile` | Ready |

## Step 1 — Commit the CI files

```bash
git add .github/workflows/ci.yml .github/workflows/pr-check.yml Makefile
git commit -m "feat: add CI/CD pipeline (GitHub Actions + Makefile)"
git push origin main
```

Visit **GitHub → Actions tab** to watch the first run.

## Step 2 — Add GitHub Secrets

Go to: **GitHub repo → Settings → Secrets and variables → Actions**

| Secret name | Where to get it | Required? |
|------------|-----------------|-----------|
| `RENDER_DEPLOY_HOOK_URL` | Render → service → Settings → Deploy Hook | Yes |
| `TALENTLENS_PRODUCTION_URL` | Your Render service URL | For verification |

## Step 3 — Verify the pipeline

```bash
# Locally — mirror what CI does
make test-fast    # same as CI test job
make lint         # CI lint gate (ruff E,F,W,I; black warns, does not fail)
make docker-build # same as CI docker job
# optional: make lint-all — full pyproject ruff; exploratory, often fails (N806 vs ML X/y/ax)

# If all green:
git push origin main  # pipeline runs automatically
```

**Lint tiers:** `make lint` matches CI (ruff E,F,W,I on book/tests/talentlens, E501 ignored; black warns only). `make lint-all` runs the full pyproject ruleset — exploratory, not a gate; N806 conflicts with ML conventions (X, y, ax).

## Step 4 — Set up branch protection

Go to: **GitHub repo → Settings → Branches → Add rule for `main`**

Recommended rules:
- [x] Require status checks to pass before merging
  - Required checks: `Test (3.11)`, `Code Quality`
- [x] Require branches to be up to date before merging
- [x] Require pull request reviews (1 approver)

This prevents anyone (including yourself) from pushing directly to
`main` without passing CI. All changes go through PRs.

## What to expect

**On every `git push origin main`:**
```
✅ Test (3.11)     ~2 min
✅ Test (3.12)     ~2 min  (parallel)
✅ Code Quality    ~40s    (parallel)
✅ Docker Build    ~5 min
✅ Deploy          ~3 min
Total: ~10 min from push to production
```

**On every pull request:**
```
✅ Tests           ~2 min
✅ Lint            ~40s
Total: ~2 min feedback on your PR
```

**If a step fails:**
- Pipeline stops — nothing broken reaches production
- GitHub sends an email notification
- Check Actions tab for the failing step and its logs
- Fix, push, pipeline re-runs automatically

## Common setup issues

**"Permission denied" on Makefile targets:**
```bash
chmod +x  # Not needed — make runs shell directly
# If the issue is on Windows: use WSL or Git Bash
```

**"No such file: .github/workflows/ci.yml":**
```bash
# Create the directories first
mkdir -p .github/workflows
# Then commit the YAML files
```

**Pipeline runs but deploy doesn't trigger:**
```bash
# Check that RENDER_DEPLOY_HOOK_URL secret is set
# Check that the deploy job has: if: github.ref == 'refs/heads/main'
# Check the deploy job's 'needs' list includes all preceding jobs
```

**Docker build fails in CI but works locally:**
```bash
# Most common causes:
# 1. Different Docker version — CI uses latest stable
# 2. Platform mismatch — CI uses linux/amd64, Mac uses arm64
# Add to Dockerfile or build args: --platform linux/amd64
```
