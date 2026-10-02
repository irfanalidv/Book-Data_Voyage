# Contributing to Data Voyage

Thank you for helping improve the book. Two kinds of contribution are especially useful:

- **Reproducibility reports.** A chapter prints a number that differs from the text, or a script fails on a fresh clone.
- **Corrections.** A factual error, a broken link, a typo, or a claim your own production experience contradicts.

## Before you open an issue

1. Run the script from the repository root, not from inside `book/chNN/`.
2. Check whether `data/clean/jobs_clean.large.csv` exists. If it does, you are on your own collected data and the numbers are expected to differ.
3. Confirm you installed from the lockfile: `make install`.

Then use the matching [issue template](https://github.com/irfanalidv/Book-Data_Voyage/issues/new/choose).

## Pull requests

Code changes are welcome under the MIT license. Before you open a pull request:

```bash
make install-dev
make test          # all tests must pass
make lint          # ruff must be clean; black is advisory
make type-check    # if you touched talentlens/
```

- Keep each pull request to one change, and say which chapter it affects.
- If a change moves a number quoted in a chapter, update the chapter text in the same pull request and say so in the description.
- To change a dependency, edit `requirements.txt`, run `make lock`, and include the new lockfile.

## Chapter text

The chapter text is © Irfan Ali, all rights reserved (see [LICENSE-BOOK.md](LICENSE-BOOK.md)). Corrections to it are welcome as issues or small pull requests. By submitting a text change you agree that it may be included in the book, in any format, without payment.

## Conduct

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
