# TOML: A Mini Guide

## Purpose

TOML (Tom's Obvious, Minimal Language) is a configuration file format designed to map unambiguously to a hash table while remaining easy for humans to read and write. It was created by Tom Preston-Werner in 2013 as a reaction to YAML's complexity and ambiguity (the spec is ~100x shorter than YAML's) and INI's lack of standardization and nesting. Spec: https://toml.io/

Its dominant role in the Python ecosystem is `pyproject.toml`, the standardized project configuration file (PEP 518 for build requirements, PEP 621 for project metadata). It is also the native config format of Rust's Cargo (`Cargo.toml`), and is used by tools like ruff, black, pytest, mypy, and uv for their configuration tables.

Design positioning relative to neighbors: JSON is unambiguous but hostile to hand-editing (no comments, strict commas); YAML is comment-friendly but has notorious ambiguities (the Norway problem: bare `no` parses as `false`); INI is simple but has no spec, no types, and one nesting level. TOML takes JSON's strict typing, adds comments and ergonomics, and bounds nesting complexity.

## Syntax

A TOML document is a set of **key/value pairs**, optionally grouped into **tables**. Whitespace is insignificant outside strings; `#` starts a comment.

### Keys and values

```toml
# bare keys: A-Za-z0-9_- only
name = "spacecore"
"quoted key" = 1          # quoted keys allow anything
physical.color = "orange"  # dotted keys create nested tables inline
```

Every value has a type, inferred from syntax — there is no untyped scalar:

```toml
# Strings
basic = "supports escapes: \n \t \u00E9"
literal = 'no escapes; good for regex and Windows paths: C:\Users'
multiline = """
line one
line two"""
multiline_literal = '''
raw text, no escaping'''

# Numbers
int = 42
hex = 0xDEADBEEF
float = 3.1415
sci = 1e-8
inf_val = inf
nan_val = nan
big = 1_000_000        # underscores for readability

# Booleans — lowercase only
enabled = true

# Dates and times (RFC 3339) — first-class types, no quotes
date = 2026-06-10
datetime = 2026-06-10T15:30:00-03:00

# Arrays — heterogeneous allowed since TOML 1.0, trailing comma OK
deps = ["numpy", "jax"]
matrix = [[1, 2], [3, 4]]

# Inline tables — single line, no trailing comma, intended for small values
point = { x = 1.0, y = 2.0 }
```

### Tables

A `[header]` defines a table; all key/value pairs after it belong to that table until the next header:

```toml
[server]
host = "0.0.0.0"
port = 8080

[server.tls]      # nested table via dotted header
enabled = true
```

**Arrays of tables** use double brackets; each `[[name]]` appends one element:

```toml
[[products]]
name = "Hammer"
sku = 738594937

[[products]]
name = "Nail"
sku = 284758393
```

### Rules worth knowing

- A key may be defined only once; redefining a table that was already created is an error.
- Tables need not appear in nesting order, but a super-table cannot be defined *after* it was implicitly created and then explicitly redefined — in practice: define `[a]` before `[a.b]` if you want keys directly in `[a]`.
- No null/None value exists; absence of a key is the only way to express "unset."
- Indentation is permitted but carries no meaning (unlike YAML).

## Standard structure: pyproject.toml

The canonical structure of the file you will encounter most often:

```toml
[build-system]                      # PEP 518 — how to build the package
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]                           # PEP 621 — standardized metadata
name = "spacecore"
version = "0.3.0"
description = "Backend-agnostic linear algebra spaces"
readme = "README.md"
requires-python = ">=3.10"
license = { file = "LICENSE" }
authors = [{ name = "Pavlo", email = "..." }]
dependencies = ["array-api-compat>=1.4"]

[project.optional-dependencies]
jax = ["jax>=0.4"]
dev = ["pytest", "ruff"]

[project.urls]
Repository = "https://github.com/..."

[tool.ruff]                         # [tool.*] — reserved namespace where
line-length = 100                   # each tool owns its own subtable

[tool.pytest.ini_options]
testpaths = ["tests"]
```

The structural convention: `[build-system]` and `[project]` are spec-defined; everything else lives under `[tool.<toolname>]`, a namespace each tool governs itself. This is why one file can configure the build, linter, formatter, and test runner without collision.

## Parsing in Python

Since Python 3.11, reading is in the standard library:

```python
import tomllib                       # read-only, stdlib (PEP 680)
with open("pyproject.toml", "rb") as f:   # must be binary mode
    cfg = tomllib.load(f)
```

Writing requires a third-party package (`tomli-w` for minimal writing, `tomlkit` for style-preserving round-trips — the right choice when programmatically editing a user's config without destroying their comments and formatting).

## References

1. TOML v1.0.0 specification — https://toml.io/en/v1.0.0
2. PEP 518 (build-system requirements) — https://peps.python.org/pep-0518/
3. PEP 621 (project metadata) — https://peps.python.org/pep-0621/
4. PEP 680 (`tomllib` in stdlib) — https://peps.python.org/pep-0680/
5. pyproject.toml guide (Python Packaging User Guide) — https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
6. tomlkit (style-preserving editing) — https://tomlkit.readthedocs.io/