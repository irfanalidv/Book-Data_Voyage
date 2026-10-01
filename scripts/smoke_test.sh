#!/usr/bin/env bash
#
# scripts/smoke_test.sh
# ---------------------
# Quick verification that:
#   1. The pinned requirements resolve cleanly in a fresh virtualenv
#   2. The 10 "complete" chapters can be imported without error
#
# This does NOT run the chapter code end-to-end (that's `make test`'s job).
# It is the cheapest possible signal that nothing has obviously rotted.
#
# Usage:
#   bash scripts/smoke_test.sh            # full smoke test (creates fresh venv)
#   bash scripts/smoke_test.sh --import   # imports only, against active venv
#
# Expected runtime: ~3 minutes for full test, ~10 seconds for import-only.

set -euo pipefail

MODE="${1:-full}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# The chapters considered "complete" — only these are smoke-tested.
# Stub chapters are skipped (they'd false-positive failures).
COMPLETE_CHAPTERS=(05 07 09 16 17 19 20 21 22 24)

run_imports() {
    echo "→ Smoke-testing imports for ${#COMPLETE_CHAPTERS[@]} complete chapters"
    local failed=0
    for ch in "${COMPLETE_CHAPTERS[@]}"; do
        # Find the chXX_*.py file in the chapter folder
        local py_file
        py_file=$(ls "book/ch${ch}/ch${ch}_"*.py 2>/dev/null | head -1)
        if [[ -z "$py_file" ]]; then
            echo "  ✗ ch${ch}: no chapter .py file found"
            failed=$((failed + 1))
            continue
        fi
        # Module path: book.ch05.ch05_data_collection (etc.)
        local module
        module=$(echo "$py_file" | sed 's|/|.|g; s|\.py$||')
        if python -c "import importlib; importlib.import_module('${module}')" 2>/dev/null; then
            echo "  ✓ ch${ch}: imports clean"
        else
            echo "  ✗ ch${ch}: import failed"
            python -c "import importlib; importlib.import_module('${module}')" 2>&1 | sed 's/^/      /' | tail -5
            failed=$((failed + 1))
        fi
    done
    echo
    if [[ $failed -eq 0 ]]; then
        echo "✓ All ${#COMPLETE_CHAPTERS[@]} complete chapters import cleanly."
        return 0
    else
        echo "✗ ${failed} chapter(s) failed import."
        return 1
    fi
}

case "$MODE" in
    --import)
        run_imports
        ;;
    full)
        echo "→ Creating fresh virtualenv at .venv-smoke"
        python -m venv .venv-smoke
        # shellcheck disable=SC1091
        source .venv-smoke/bin/activate
        echo
        echo "→ Upgrading pip"
        pip install --quiet --upgrade pip
        echo
        echo "→ Installing requirements.txt (this takes 2-3 min on first run)"
        pip install --quiet -r requirements.txt
        echo "✓ requirements.txt resolved and installed"
        echo
        run_imports
        echo
        echo "→ Cleaning up: deactivate and remove .venv-smoke"
        deactivate
        rm -rf .venv-smoke
        ;;
    *)
        echo "Usage: $0 [--import]"
        exit 1
        ;;
esac
