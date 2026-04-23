# Quiver FlashANNS Branch Refresh Design

## Goal

Refresh the local `huancheng/flashanns-result-ids` working branch with the latest relevant upstream `origin/main` FlashANNS fixes, preserve the ann-benchmarks integration work already carried on the result-ids branch, and rerun focused validation to see which FlashANNS issues remain.

## Context

The current local Quiver workspace contains three kinds of changes:

- the branch tip shared with `origin/huancheng/flashanns-result-ids`
- local uncommitted FlashANNS debugging changes for invalid entry ids, page bounds checks, and diagnostics
- newer upstream `origin/main` commits that changed FlashANNS and nearby Quiver pipeline code after the result-ids branch diverged

The current runtime state is partially functional:

- `100` query native FlashANNS search completes
- `10000` query native FlashANNS search stalls later in the run
- current output quality is still suspicious because the sample output contains `inf(...)`
- current memory-backend throughput is not yet comparable to a healthy baseline

## Requirements

The refresh flow must preserve the existing ann-benchmarks integration branch shape while incorporating the upstream fixes most likely to address:

- pipeline hang / slot retirement issues
- FlashANNS kernel-structure drift from upstream
- local entry-id and memory-loader debugging work that already produced useful evidence

## Approach Options

### Option 1: Cherry-pick only relevant upstream commits onto the result-ids branch

Create a local `huancheng/flashanns-result-ids` branch from the current result-ids tip, keep local uncommitted debugging work, and selectively cherry-pick the upstream commits that touch FlashANNS or directly explain the observed stalls.

This keeps the branch focused and minimizes conflict scope.

### Option 2: Rebase the result-ids branch onto current `origin/main`

This produces a cleaner long-term history, but it broadens conflict scope and mixes docs and unrelated Quiver changes into the current debugging session.

### Option 3: Merge all of `origin/main` into the result-ids branch

This is easy mechanically, but it drags in unrelated paper docs and Quiver changes that dilute the immediate debugging signal.

## Recommendation

Use Option 1.

This session needs a controlled integration surface with fast feedback. Selective cherry-picks give the best chance of preserving the ann-benchmarks branch semantics while still absorbing the upstream fixes most likely to matter for FlashANNS correctness and stall behavior.

## Selected Upstream Inputs

The upstream commits most relevant to the current issues are:

- `1819760` `refactor(flashanns): align kernel with VLDB 2026 paper design`
- `e74593a` `quiver: fix Q-interleave hang by retiring exhausted slots`
- `6aa0c57` `quiver: optimize pipe loop to reduce p50 latency ~2x`

The first commit touches `src/flashanns/*` directly. The second and third commits change `src/quiver/*`, so they should be treated as design references first and ported into FlashANNS only where the logic matches the current bug.

## Branch Strategy

### Local branch target

Create or switch to local branch `huancheng/flashanns-result-ids`.

This branch will continue to carry:

- the existing `feat: export flashanns result ids` integration work
- the local debugging patches already made in this session
- the selected upstream refresh work

### Preservation of local work

Preserve current uncommitted local debugging changes before any branch surgery. The cleanest approach is:

1. switch to the local `huancheng/flashanns-result-ids` branch at the current result-ids tip
2. commit or stage the current local diagnostic work on that branch
3. cherry-pick upstream refresh commits and resolve conflicts in place

## Integration Scope

### Keep directly

- current local diagnostics and tests
- current local memory-loader bounds guards
- current local `nav entry` validation path
- current result-ids branch CLI/export behavior

### Import directly or adapt

- upstream FlashANNS kernel-structure changes from `1819760`
- any slot-retirement logic from `e74593a` that maps cleanly onto FlashANNS lane scheduling

### Keep out for now

- paper/docs-only commits
- unrelated Quiver experiments that do not inform FlashANNS memory backend correctness or stall behavior

## Validation Strategy

### Step 1: Compile-time validation

Rebuild FlashANNS test and search targets:

- `flashanns_node_validation_test`
- `flashanns_nav_entry_test`
- `flashanns_search`

### Step 2: Correctness smoke tests

Run:

- `flashanns_node_validation_test`
- `flashanns_nav_entry_test --max-queries 10000`
- native `flashanns_search` on `100` queries

The `100` query run must complete with finite distances in the printed sample output.

### Step 3: Stall validation

Run native `flashanns_search` on a larger query prefix and scale upward:

- `1000`
- `2000`
- `5000`
- `10000`

This isolates the first failing prefix quickly and makes the hang reproducible.

### Step 4: ann-benchmarks verification

After native validation passes, rerun the ann-benchmarks FlashANNS path only.

## Success Criteria

The refreshed branch is acceptable when all of the following are true:

- local branch `huancheng/flashanns-result-ids` contains the result-ids integration work plus the selected upstream refresh work
- native `flashanns_search` completes on `10000` queries without hanging
- printed sample results contain finite distances
- ann-benchmarks FlashANNS run completes with the refreshed binary

