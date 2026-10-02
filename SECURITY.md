# Security policy

## Reporting a vulnerability

Please do not report security problems in public issues. Email **irfan@datacortex.in** with:

- what you found and where (file, endpoint, or workflow),
- the steps to reproduce it,
- the impact you expect.

You will get an acknowledgement within five working days. Confirmed problems are fixed in this repository and noted in [CHANGELOG.md](CHANGELOG.md). Reporters are credited unless they prefer otherwise.

## Scope

This repository is teaching code. The TalentLens API (Chapters 19 and 20) is a worked example, not a hosted service, so treat it as a starting point and harden it before you expose it to the internet.

In scope: the code in `book/`, `talentlens/`, `scripts/`, the `Dockerfile`, and the GitHub Actions workflows.

Known advisories in the pinned PyTorch and Transformers are explained in [SCOPE.md](SCOPE.md#pinned-dependencies), with what they cover and how to upgrade for production. Reports about other dependencies are welcome.

## Secrets

The repository never needs a secret to run. API keys for Adzuna, Groq or OpenAI belong in `.env`, which git ignores. If you find a secret committed anywhere in the history, report it as above.
