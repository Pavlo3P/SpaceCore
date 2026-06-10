#!/usr/bin/env bash
set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "error: GitHub CLI 'gh' is required to set up labels" >&2
  exit 1
fi

echo "Creating/updating SpaceCore GitHub label taxonomy..."

gh label create "good-first-issue" --force \
  --description "Beginner-safe: exact file, example, and done condition provided" \
  --color "7057ff"

gh label create "help-wanted" --force \
  --description "Well-specified work where maintainer help is welcome" \
  --color "008672"

gh label create "needs-design" --force \
  --description "Design direction is not settled; not ready for new contributors" \
  --color "d93f0b"

gh label create "documentation" --force \
  --description "Documentation work" \
  --color "0075ca"

gh label create "test" --force \
  --description "Tests and test infrastructure" \
  --color "1d76db"

gh label create "fix" --force \
  --description "Bug fix or correctness fix" \
  --color "d73a4a"

gh label create "feature" --force \
  --description "New feature or API extension" \
  --color "a2eeef"

gh label create "backend" --force \
  --description "BackendOps, Context, Array API, optional backend behavior" \
  --color "5319e7"

gh label create "space" --force \
  --description "Spaces, elements, inner products, capabilities, checks" \
  --color "5319e7"

gh label create "linop" --force \
  --description "Linear operators, adjoints, matrix-backed and matrix-free operators" \
  --color "5319e7"

gh label create "functional" --force \
  --description "Functionals, gradients, pull-backs" \
  --color "5319e7"

gh label create "linalg" --force \
  --description "Iterative algorithms and solver infrastructure" \
  --color "5319e7"

gh label create "docs" --force \
  --description "Documentation site, tutorials, developer docs" \
  --color "0075ca"

gh label create "ci" --force \
  --description "Continuous integration, release checks, automation" \
  --color "c5def5"

gh label create "release" --force \
  --description "Release preparation, changelog, tagging, packaging" \
  --color "bfdadc"

echo "Done."
