#!/usr/bin/env bash
# Runs one hourly "catch-up" batch for the permanent, Claude-Routine-driven
# paper trading run: restores prior state from the bot-state branch, runs a
# bounded batch of real-data cycles, then persists state back.
#
# Assumes it's invoked from inside an already-cloned checkout of the LMTrade
# repo, on the code branch to run (the Routine's prompt handles cloning /
# checking out the right branch before calling this script).
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

STATE_BRANCH="bot-state"
STATE_FILE="state/lmtrade.db"
CYCLES="${LMTRADE_HOURLY_CYCLES:-20}"
INTERVAL="${LMTRADE_HOURLY_INTERVAL:-60}"

echo "==> Fetching $STATE_BRANCH..."
git fetch origin "$STATE_BRANCH"

WORKTREE_DIR="$(mktemp -d)"
cleanup() { git worktree remove "$WORKTREE_DIR" --force >/dev/null 2>&1 || true; }
trap cleanup EXIT

git worktree add "$WORKTREE_DIR" "origin/$STATE_BRANCH" >/dev/null

mkdir -p data
if [ -f "$WORKTREE_DIR/$STATE_FILE" ]; then
    cp "$WORKTREE_DIR/$STATE_FILE" data/lmtrade.db
    echo "==> Restored prior state ($(du -h data/lmtrade.db | cut -f1))"
else
    echo "==> No prior state found on $STATE_BRANCH — starting fresh"
fi

echo "==> Installing..."
pip install -e . -q

echo "==> Running $CYCLES cycles at ${INTERVAL}s interval (~$((CYCLES * INTERVAL / 60)) min)..."
lmtrade run --cycles "$CYCLES" --interval "$INTERVAL"

echo "==> Persisting state..."
mkdir -p "$WORKTREE_DIR/state"
cp data/lmtrade.db "$WORKTREE_DIR/$STATE_FILE"
(
    cd "$WORKTREE_DIR"
    git add "$STATE_FILE"
    if git diff --cached --quiet; then
        echo "==> No state changes to commit"
    else
        git -c user.name="LMTrade Bot" -c user.email="bot@lmtrade.local" \
            commit -q -m "Hourly state update: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        git push origin "$STATE_BRANCH"
        echo "==> State pushed"
    fi
)

echo
echo "==> Final status:"
lmtrade status
