# ADR-012: Jordan spectrum

## Status

Accepted

## Context

Spectral operations are mathematical capabilities of Jordan algebra spaces, not generic dense-array utilities.

## Current design

`JordanAlgebraSpace` exposes `jordan`, `spectrum`, `spectral_decompose`, `from_spectrum`, and `spectral_apply`. Elementwise Jordan spaces return coordinates as their spectrum with `frame=None`. `HermitianSpace` uses backend `eigh` internally but exposes the result through the Jordan spectral API. Product Jordan spaces concatenate component spectra and return `ProductSpectralDecomposition` for structured component spectral data. Stacked Jordan spaces apply base spectral operations over the leading stack.

## Decision

Public spectral behavior belongs to space capabilities (`spectrum` and `spectral_decompose`), not to callers reaching directly for backend `eigh` except inside the implementing space.

## Rationale

Different Jordan spaces have different spectral meaning. A Hermitian matrix spectrum, an elementwise spectrum, and a product spectrum have different frames and reconstruction rules.

## Alternatives considered

Exposing `eigh` as the universal space-level operation was rejected because it only applies to Hermitian matrix coordinates. Returning raw flattened eigenpairs for products was rejected because it loses component frames and structure.

## Consequences

Only Jordan-capable spaces should expose spectral operations. Product spectral decompositions must remain structured objects. New spectral spaces must define reconstruction through `from_spectrum`. See [ADR-005](005_space_subclasses_and_capabilities.md).

## Contributor invariants

- Do not call backend `eigh` as a generic replacement for `space.spectrum`.
- `spectral_decompose` must return enough data for `from_spectrum` to reconstruct an element.
- Product spectral data must preserve component boundaries.
- Capability dispatch controls whether spectral methods are present.
