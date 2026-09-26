# How Contributor Infrastructure Is Organized: A Field Guide

This guide describes how mature open-source projects organize the scaffolding that lets outsiders contribute effectively — the documents, conventions, automation, and governance that together form "contributor infrastructure." It is written from the perspective of a maintainer preparing a scientific/technical Python library for external contributors, but the structure is general.

---

## 1. The layered model

Contributor infrastructure is best understood as four layers, ordered from "what a drive-by contributor touches first" to "what only long-term collaborators ever read":

1. **Entry layer** — README, LICENSE, CODE_OF_CONDUCT, CONTRIBUTING. These answer: *what is this, may I use it, how do I behave, how do I start?*
2. **Process layer** — issue/PR templates, labels, CI checks, review conventions, changelog and release process. These answer: *what happens when I actually submit something?*
3. **Knowledge layer** — architecture docs, Architecture Decision Records (ADRs), vision/roadmap documents, developer guides. These answer: *why is the code the way it is, and where is it going?*
4. **Governance layer** — maintainer roles, decision-making process, CODEOWNERS, security policy. These answer: *who decides, and how?*

Small projects need layers 1–2 fully and layer 3 partially; layer 4 can stay implicit until there are multiple maintainers. The most common failure mode is over-investing in layer 4 (elaborate governance documents for a one-maintainer project) while leaving layer 3 empty, so contributors cannot understand design rationale and submit PRs that fight the architecture.

**Reference:** GitHub's "community profile" checklist formalizes layer 1 — it checks for README, LICENSE, CODE_OF_CONDUCT, CONTRIBUTING, issue templates, and a security policy. See https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions

---

## 2. The entry layer

### README.md

The README is the funnel mouth. Convention (well summarized by the "standard-readme" spec and Make a README) is roughly: one-sentence description → badges (CI, coverage, package version) → install → minimal working example → links outward to deeper docs. The key organizational principle is that the README *links* to contributor infrastructure rather than *containing* it — a README that embeds the full contributing guide goes stale.

- Make a README: https://www.makeareadme.com/
- standard-readme spec: https://github.com/RichardLitt/standard-readme

### CONTRIBUTING.md

This is the canonical entry point for contributors, and GitHub surfaces it automatically when someone opens an issue or PR. A well-organized CONTRIBUTING.md is structured by *contributor journey stage*, not by topic:

1. **Before you start** — scope of the project (what kinds of contributions are wanted/unwanted), where to ask questions, whether to open an issue before a PR.
2. **Environment setup** — clone, install dev dependencies, run the test suite. This section should be copy-pasteable and verified in CI (the "fresh clone to green tests" path is itself testable).
3. **Making a change** — branch naming, commit conventions, code style, how to run linters/formatters locally.
4. **Submitting** — PR expectations, what reviewers look for, CI requirements, who merges.
5. **After merge** — changelog/release expectations, attribution.

The most-cited general resource here is GitHub's Open Source Guides, particularly "Best Practices for Maintainers" and "How to Contribute": https://opensource.guide/

A widely studied scientific-Python example is NumPy's contributor documentation, which splits the short CONTRIBUTING entry point from a longer "development workflow" guide in the docs: https://numpy.org/devdocs/dev/index.html — this two-tier split (short file in repo root, long guide in rendered docs) is the dominant pattern in scientific Python (NumPy, SciPy, scikit-learn all use it).

### LICENSE and CODE_OF_CONDUCT.md

License choice is infrastructure in the sense that it gates corporate contributors (many employers whitelist MIT/BSD/Apache-2.0 only). See https://choosealicense.com/. For codes of conduct, the de facto standard is the Contributor Covenant (https://www.contributor-covenant.org/), used by the majority of large projects; its practical function is less about behavior policing and more about signaling that the project is professionally run.

---

## 3. The process layer

### Issue and PR templates

GitHub supports structured issue forms (YAML) under `.github/ISSUE_TEMPLATE/`, plus a `config.yml` that can disable blank issues and route questions to Discussions. The organizing principle: one template per *triage outcome*. Typical taxonomy is bug report, feature request, documentation issue, and (for numerical libraries) a "numerical correctness / unexpected result" template that asks for a minimal reproducer, backend, dtype, and expected-vs-actual values. The template's job is to front-load the information a maintainer would otherwise request in the first reply.

PR templates (`.github/PULL_REQUEST_TEMPLATE.md`) work best as checklists: linked issue, tests added, changelog entry, docs updated. Reference: https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests

### Label taxonomy

Labels serve three distinct functions, and well-organized projects keep the dimensions orthogonal rather than mixing them in one flat list:

1. **Type** — bug, enhancement, documentation, maintenance.
2. **Status/triage** — needs-triage, needs-reproducer, confirmed, blocked, wontfix.
3. **Contributor routing** — good first issue, help wanted, needs-design-discussion.

"good first issue" is special: GitHub indexes it for discovery, and its real function is curation — a maintainer pre-decomposes a task so that a newcomer can succeed without architectural context. The discipline of writing good-first-issues (clear acceptance criteria, pointers to the relevant module, an explicit non-goal list) is essentially the same skill as writing a phased task specification. Many projects also adopt area/component labels (e.g. `area:linalg`, `area:ci`) once the codebase has enough modules that triage benefits from routing.

Reference on label conventions and triage practices: Kubernetes' contributor guide is the most elaborated public example — https://www.kubernetes.dev/docs/guide/ — though for a small project the lesson is to adopt its *dimensions*, not its scale.

### CI as enforced contract

Continuous integration is the part of contributor infrastructure that replaces human nagging with machine feedback. The mature organization is: every requirement stated in CONTRIBUTING.md should be either (a) enforced by CI or (b) explicitly marked as judgment-based. Formatting, linting, type checking, and test passes belong in (a); design quality belongs in (b). Pre-commit hooks (https://pre-commit.com/) mirror the CI checks locally so contributors get feedback before pushing. The anti-pattern is a CONTRIBUTING.md that asks for things CI doesn't check — these requirements decay immediately.

### Commit and changelog conventions

Two interlocking standards dominate. **Conventional Commits** (https://www.conventionalcommits.org/) gives commit messages a machine-parseable shape (`feat:`, `fix:`, `refactor:`), which enables automated changelog generation and semantic-version inference. **Keep a Changelog** (https://keepachangelog.com/) defines the human-facing CHANGELOG.md format (Added / Changed / Deprecated / Removed / Fixed / Security sections per release). **Semantic Versioning** (https://semver.org/) ties releases to API-compatibility promises — which matters acutely for a library, since contributors need to know whether their change is allowed to break public API in the next release.

Some projects (notably pip, pytest, and Twisted) instead use *news fragments*: each PR adds a small file under `changes/` describing its effect, and a tool (towncrier, https://towncrier.readthedocs.io/) assembles the changelog at release time. This avoids the perennial merge conflict on CHANGELOG.md and makes "changelog entry present" a CI-checkable requirement.

---

## 4. The knowledge layer

This is the layer that distinguishes projects where external contributions *converge* with the architecture from projects where every PR is a negotiation.

### Architecture Decision Records (ADRs)

An ADR is a short document capturing one significant design decision: the context, the decision, the alternatives considered, and the consequences. The canonical formulation is Michael Nygard's 2011 post "Documenting Architecture Decisions" (https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions), and the standard template lives at https://adr.github.io/ (see also the MADR template, https://adr.github.io/madr/).

Organizational conventions that have proven durable:

- ADRs live in-repo (commonly `docs/adr/` or `docs/decisions/`), numbered sequentially (`0001-record-architecture-decisions.md`), and are immutable once accepted — a reversed decision gets a *new* ADR that supersedes the old one, preserving the historical record.
- Each ADR has a status: proposed → accepted → superseded/deprecated.
- ADRs document decisions, not designs. "We use a capability hierarchy rather than feature flags on a monolithic Space class, because X, at the cost of Y" is an ADR; a class diagram is not.

For a library, the highest-value ADRs are the ones a contributor would otherwise re-litigate: backend-abstraction strategy, public-API surface philosophy, dtype/precision policy, why certain seemingly-obvious abstractions were deliberately *not* built. Negative decisions ("we considered and rejected Z") are disproportionately valuable because rejected designs get re-proposed by newcomers indefinitely unless the rejection is written down.

### Vision and roadmap documents

A vision document states what the project is and is not for; a roadmap states sequencing. The functional distinction: the vision document lets a maintainer decline out-of-scope contributions by pointing at a paragraph rather than arguing case-by-case, and the roadmap lets contributors pick work that will actually be merged. NumPy's NEP process (https://numpy.org/neps/) and Python's PEPs (https://peps.python.org/pep-0001/) are the heavyweight versions; for small projects a single `docs/vision.md` plus milestone-organized issues serve the same function.

### Developer/architecture guides

Distinct from user documentation: a `docs/architecture.md` or "internals" guide that gives a map of the modules, the core invariants ("every Space subclass must satisfy these properties; the test suite enforces them via this shared check module"), and where new code of each kind should go. The rule of thumb from the Open Source Guides: document the things you find yourself explaining twice.

---

## 5. The governance layer

For single-maintainer projects this layer is mostly latent, but two pieces are worth installing early because they are cheap and load-bearing:

**CODEOWNERS** (`.github/CODEOWNERS`) maps paths to required reviewers. With one maintainer it is trivial, but it establishes the routing mechanism before it is needed. Reference: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners

**SECURITY.md** states how to report vulnerabilities privately. GitHub's community profile checks for it, and its absence means security reports arrive as public issues.

When projects grow, governance formalizes along a known spectrum: BDFL (single benevolent maintainer with final say) → meritocratic committer model (Apache-style) → formal steering councils (Python post-PEP-13, NumPy). The Open Source Guides chapter on leadership and governance (https://opensource.guide/leadership-and-governance/) surveys these. The practical takeaway for a small project: write down the *current* model in one paragraph in CONTRIBUTING.md ("decisions are made by the maintainer; substantial design changes should be proposed as an issue before implementation") so that the implicit becomes explicit.

A related convention is **attribution infrastructure**: the All Contributors specification (https://allcontributors.org/) recognizes non-code contributions (docs, triage, design), and CITATION.cff (https://citation-file-format.github.io/) gives academic users a citable reference — particularly relevant for scientific software, where citation is the currency contributors and users care about.

---

## 6. How the pieces interlock

The infrastructure is a pipeline, and its quality is determined by the seams:

1. A user hits a problem → the **issue template** extracts a reproducer → **labels** triage it → if suitable, it becomes a **good first issue** with a decomposed task description.
2. A contributor reads **CONTRIBUTING.md** → sets up the environment (verified by the same commands **CI** runs) → consults the **architecture guide and ADRs** to place the change correctly.
3. The **PR template** checklist plus **CI** enforce the mechanical requirements → review covers only judgment → the **changelog fragment** merges with the PR.
4. **Conventional commits / semver** determine the release → the **changelog** assembles → release notes credit contributors.

Every document in the system should be reachable from this pipeline; a document no pipeline stage points to will not be read. Conversely, every recurring maintainer intervention ("please add a test", "this was already rejected in 2024", "wrong module for this") signals a missing or unlinked piece of infrastructure.

---

## 7. Minimal viable set, in adoption order

For a project moving from solo development to accepting contributors, the empirically sensible order is:

1. LICENSE, README with working install + example (already assumed).
2. CONTRIBUTING.md with a verified fresh-clone-to-green-tests path.
3. CI enforcing format/lint/type/test; pre-commit config mirroring it.
4. Issue templates + a small orthogonal label set + a config.yml routing questions away from issues.
5. ADRs for the decisions you have already made and would have to defend — written retroactively, which is legitimate and common (Nygard's template explicitly supports recording past decisions).
6. PR template + changelog convention (towncrier-style fragments scale best).
7. CODE_OF_CONDUCT.md, SECURITY.md, CITATION.cff.
8. Vision doc + curated good-first-issues, last, because they only pay off once steps 1–7 make the contribution path frictionless.

---

## References

1. GitHub Open Source Guides — https://opensource.guide/ (esp. "Best Practices for Maintainers", "Building Community", "Leadership and Governance")
2. GitHub Docs, "Setting up your project for healthy contributions" — https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions
3. GitHub Docs, issue/PR templates — https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests
4. Michael Nygard, "Documenting Architecture Decisions" (2011) — https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions
5. ADR organization and templates — https://adr.github.io/ and MADR: https://adr.github.io/madr/
6. Conventional Commits — https://www.conventionalcommits.org/
7. Keep a Changelog — https://keepachangelog.com/
8. Semantic Versioning — https://semver.org/
9. towncrier (news-fragment changelogs) — https://towncrier.readthedocs.io/
10. pre-commit — https://pre-commit.com/
11. Contributor Covenant — https://www.contributor-covenant.org/
12. choosealicense — https://choosealicense.com/
13. NumPy contributor & development docs — https://numpy.org/devdocs/dev/index.html ; NEP process: https://numpy.org/neps/
14. PEP 1 (proposal-process archetype) — https://peps.python.org/pep-0001/
15. Kubernetes contributor guide (large-scale reference) — https://www.kubernetes.dev/docs/guide/
16. All Contributors specification — https://allcontributors.org/
17. Citation File Format (CITATION.cff) — https://citation-file-format.github.io/
18. CODEOWNERS — https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners