# Contributor prerequisites

SpaceCore does not abstract away the mathematics. It gives mathematical structures to objects. A geometrically incorrect contribution that passes tests is still wrong; mathematical correctness is part of review.

| Contribution type | Mathematical prerequisite | Pointer |
| --- | --- | --- |
| Adding or fixing a docstring | None | `setup.md` |
| Adding a test for existing behavior | Basic NumPy and pytest | `setup.md` |
| New check in `_checks.py` | Linear algebra: shapes, dtypes, validation cost | ADR-014 |
| Dtype or scalar-field behavior | Scalar fields, dtype promotion, backend array semantics | ADR-015 |
| New `InnerProduct` subclass | Inner product spaces and Riesz maps | ADR-004 and ADR-009 |
| New matrix-backed `LinOp` | Linear maps and adjoints | LinOp ADRs |
| New matrix-free `LinOp` | Linear maps, metric adjoints, and algorithm knowledge | LinOp ADRs |
| New linalg method | Algorithm theory and convergence assumptions | Must cite a reference in PR |
| Extending Jordan hierarchy | Jordan algebras and spectral calculus | ADR-012 |
| New Space type | Functional analysis and all relevant lower-level contracts | Architecture ADRs |

Use the lightest prerequisite that is honest for the change. Documentation-only and test-only contributions should not require solver theory. Geometry, adjoint, scalar-field, batching, and spectral changes require explicit mathematical reasoning in the PR because tests can miss invalid assumptions.

Useful ADR entry points:

- ADR-003 for space hierarchy and coordinate representation.
- ADR-004 for inner products and geometry.
- ADR-007 and ADR-008 for LinOp contracts and subclasses.
- ADR-009 for metric adjoints and Riesz maps.
- ADR-011 for linalg solver contracts.
- ADR-012 for Jordan spectral behavior.
- ADR-014 for validation/check policy.
- ADR-015 for dtype defaults and scalar-field planning.
