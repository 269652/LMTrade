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
LEDGER_FILE="state/ledger.csv"
# 12 cycles x ~30-symbol universe (concurrent, bounded fetch) keeps total
# Yahoo request volume reasonable per hour — the universe grew 6x (5 -> 30
# symbols) since this default was first set to 20; sustained high request
# volume is what triggered rate-limiting in the incident that led to the
# data-source-safety fix (a rate-limited symbol degrades safely now, but
# there's no reason to court it more than necessary).
CYCLES="${LMTRADE_HOURLY_CYCLES:-12}"
INTERVAL="${LMTRADE_HOURLY_INTERVAL:-60}"

echo "==> Fetching $STATE_BRANCH..."
git fetch origin "$STATE_BRANCH"

WORKTREE_DIR="$(mktemp -d)"
cleanup() { git worktree remove "$WORKTREE_DIR" --force >/dev/null 2>&1 || true; }
trap cleanup EXIT

git worktree add -B "$STATE_BRANCH" "$WORKTREE_DIR" "origin/$STATE_BRANCH" >/dev/null

mkdir -p data
if [ -f "$WORKTREE_DIR/$STATE_FILE" ]; then
    cp "$WORKTREE_DIR/$STATE_FILE" data/lmtrade.db
    echo "==> Restored prior state ($(du -h data/lmtrade.db | cut -f1))"
else
    echo "==> No prior state found on $STATE_BRANCH — starting fresh"
fi

echo "==> Installing..."
pip install -e . -q

# Optional signal imports, prepared by the invoking Claude Routine session:
# LMTRADE_NEWS_FILE     — hourly market news:   [{symbol, text, sentiment?}]
# LMTRADE_ANALYSIS_FILE — daily market analysis: {valid_hours, symbols: {...}}
# Imported AFTER state restore so they land in the live DB the engine reads.
if [ -n "${LMTRADE_NEWS_FILE:-}" ] && [ -f "${LMTRADE_NEWS_FILE}" ]; then
    echo "==> Importing news from ${LMTRADE_NEWS_FILE}..."
    lmtrade import-news "${LMTRADE_NEWS_FILE}"
fi
if [ -n "${LMTRADE_ANALYSIS_FILE:-}" ] && [ -f "${LMTRADE_ANALYSIS_FILE}" ]; then
    echo "==> Importing analysis from ${LMTRADE_ANALYSIS_FILE}..."
    lmtrade import-analysis "${LMTRADE_ANALYSIS_FILE}"
fi

echo "==> Running $CYCLES cycles at ${INTERVAL}s interval (~$((CYCLES * INTERVAL / 60)) min)..."
lmtrade run --cycles "$CYCLES" --interval "$INTERVAL"

echo "==> Persisting state..."
mkdir -p "$WORKTREE_DIR/state"
cp data/lmtrade.db "$WORKTREE_DIR/$STATE_FILE"

# Diffable trade ledger — reviewable via git history, unlike the opaque
# SQLite file. Incremental: only appends trades not yet exported.
lmtrade export-ledger "$WORKTREE_DIR/$LEDGER_FILE"

(
    cd "$WORKTREE_DIR"
    git add "$STATE_FILE" "$LEDGER_FILE"
    if git diff --cached --quiet; then
        echo "==> No state changes to commit"
    else
        git -c user.name="LMTrade Bot" -c user.email="bot@lmtrade.local" \
            commit -q -m "Hourly state update: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        # Another firing may have pushed since this worktree was created from
        # origin/$STATE_BRANCH ~20 min ago — rebase onto the latest before
        # pushing so we never silently clobber a concurrent update.
        git pull --rebase origin "$STATE_BRANCH"
        git push origin "$STATE_BRANCH"
        echo "==> State pushed"
    fi
)

echo
echo "==> Final status:"
lmtrade status
