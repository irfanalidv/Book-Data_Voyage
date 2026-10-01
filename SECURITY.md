# Security policy

## Reporting a vulnerability

Please do not report security problems in public issues. Email **irfan@datacortex.in** with:

- what you found and where (file, endpoint, or workflow),
- the steps to reproduce it,
- the impact you expect.

You will get an acknowledgement within five working days. Once a fix is ready, it ships in the next release and is noted in [CHANGELOG.md](CHANGELOG.md). Reporters are credited unless they prefer otherwise.

## Scope

This repository is teaching code. The TalentLens API (Chapters 19 and 20) is a worked example, not a hosted service, so treat it as a starting point and harden it before you expose it to the internet.

In scope: the code in `book/`, `talentlens/`, `scripts/`, the `Dockerfile`, and the GitHub Actions workflows.

Known advisories in pinned third-party packages (PyTorch, Transformers) are tracked in [ROADMAP.md](ROADMAP.md), since upgrading them changes numbers quoted in the book. Reports about other dependencies are welcome.

## Secrets

The repository never needs a secret to run. API keys for Adzuna, Groq or OpenAI belong in `.env`, which git ignores. If you find a secret committed anywhere in the history, report it as above.
