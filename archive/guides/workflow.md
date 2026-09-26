# GitHub Workflows: A Mini Guide

## Purpose

A GitHub workflow is an automated process defined in a YAML file under `.github/workflows/` that GitHub Actions executes in response to repository events. Workflows are the enforcement mechanism of contributor infrastructure: tests, linting, type checking, packaging, releases, and issue triage become machine-checked rather than human-policed.

The platform's conceptual hierarchy, top-down:

**Workflow** → triggered by **events** → contains one or more **jobs** (each on a fresh runner VM, parallel by default) → each job runs a sequence of **steps** (a shell command or a reusable **action**).

Docs root: https://docs.github.com/en/actions

## Anatomy of a workflow file

```yaml
name: CI                          # display name in the Actions tab

on:                               # events that trigger this workflow
  push:
    branches: [main]
  pull_request:                   # runs on PRs targeting any branch
  workflow_dispatch:              # manual "Run workflow" button

jobs:
  test:
    runs-on: ubuntu-latest        # runner image
    strategy:
      matrix:                     # fan out: one job per combination
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4           # an action (reusable step)
      - uses: actions/setup-python@v5
        with:                               # action inputs
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"        # a shell command
      - run: pytest
```

Key semantics:

- **Jobs are isolated.** Each job gets a fresh VM; nothing persists between jobs unless explicitly passed via artifacts (`actions/upload-artifact` / `download-artifact`) or job `outputs`. Steps within a job share the filesystem.
- **Jobs run in parallel** unless ordered with `needs: [other-job]`, which builds a DAG.
- **`uses:` vs `run:`** — `uses:` invokes a packaged action (from the marketplace or a path in the repo); `run:` executes shell. Pin marketplace actions at least to a major tag (`@v4`); security-sensitive projects pin to a full commit SHA, since tags are mutable.
- **Expressions** use `${{ ... }}` syntax with contexts: `github` (event payload, ref, actor), `matrix`, `secrets`, `env`, `steps.<id>.outputs`, `needs.<job>.outputs`.

## The pieces that matter in practice

### Triggers (`on:`)

The workhorses: `push`, `pull_request`, `workflow_dispatch` (manual), `schedule` (cron), `release` (on publishing a GitHub release), and `workflow_call` (makes the workflow reusable from other workflows). Filters narrow them — `branches:`, `tags:`, and `paths:` (e.g. skip CI when only docs changed: `paths-ignore: ["docs/**", "*.md"]`).

A subtle but important distinction: `pull_request` from a fork runs with read-only permissions and no secrets (so malicious PRs can't exfiltrate them); `pull_request_target` runs with secrets in the base repo's context and is a well-known footgun if it checks out untrusted PR code. Reference: https://securitylab.github.com/resources/github-actions-preventing-pwn-requests/

### Matrix builds

`strategy.matrix` is the standard way a library tests across Python versions, operating systems, and dependency variants. `include:` adds specific extra combinations; `exclude:` removes them; `fail-fast: false` lets all combinations finish even when one fails (usually what you want for diagnosing which backend/version broke).

```yaml
strategy:
  fail-fast: false
  matrix:
    os: [ubuntu-latest, macos-latest]
    python-version: ["3.10", "3.12"]
```

### Caching

`actions/cache` (or the built-in `cache:` input of `setup-python`) keys a cache on a hash of lockfiles/requirements, cutting dependency-install time. Cache keys are the usual source of staleness bugs — include the OS, Python version, and dependency-file hash in the key.

### Permissions and secrets

Each workflow gets an automatic `GITHUB_TOKEN`; its scope should be minimized at the top of the file:

```yaml
permissions:
  contents: read
```

and broadened per job only where needed (e.g. `contents: write` for a release job, `id-token: write` for PyPI Trusted Publishing). Repository secrets live in Settings → Secrets and are referenced as `${{ secrets.NAME }}` — never echoed, never available to fork PRs.

### Concurrency

Cancel superseded runs on the same PR to save runner minutes:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

### Reusable workflows and composite actions

Two deduplication mechanisms: a **reusable workflow** (`on: workflow_call`) is invoked as a whole job from another workflow; a **composite action** (an `action.yml` bundling several steps) is invoked as a single step. Rule of thumb: composite action for a repeated step sequence within jobs (setup + install), reusable workflow for a repeated job topology across repos.

## Standard structure for a Python library

The conventional file split — one workflow per lifecycle concern:

```
.github/workflows/
  ci.yml         # on: push/pull_request — lint, typecheck, test matrix
  docs.yml       # build docs; deploy on main
  release.yml    # on: release published — build sdist/wheel, publish to PyPI
```

A modern `release.yml` uses **PyPI Trusted Publishing** (OIDC, no long-lived API token stored as a secret):

```yaml
jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - run: pipx run build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

Reference: https://docs.pypi.org/trusted-publishers/

Branch protection (Settings → Branches, or rulesets) is the closing link: marking CI jobs as **required status checks** is what turns a workflow from advisory into enforced contract — a contributor's PR cannot merge red.

## Debugging and ergonomics

- Re-run with debug logging: set secret `ACTIONS_STEP_DEBUG=true`, or use the "Re-run with debug logging" button.
- `act` (https://github.com/nektos/act) runs workflows locally in Docker — approximate but useful for iteration.
- Step `id:` plus `>> "$GITHUB_OUTPUT"` passes values between steps; `>> "$GITHUB_ENV"` sets env vars for subsequent steps.
- `if:` conditions gate steps/jobs: `if: github.event_name == 'push'`, `if: failure()` for cleanup/notification steps.

## References

1. GitHub Actions documentation — https://docs.github.com/en/actions
2. Workflow syntax reference — https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions
3. Events that trigger workflows — https://docs.github.com/en/actions/reference/events-that-trigger-workflows
4. Contexts and expressions — https://docs.github.com/en/actions/reference/contexts-and-expression-syntax-for-github-actions
5. Security hardening for GitHub Actions — https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions
6. "Preventing pwn requests" (pull_request_target pitfalls) — https://securitylab.github.com/resources/github-actions-preventing-pwn-requests/
7. PyPI Trusted Publishers — https://docs.pypi.org/trusted-publishers/
8. Python Packaging Guide: publishing with Actions — https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/
9. act (local workflow runner) — https://github.com/nektos/act