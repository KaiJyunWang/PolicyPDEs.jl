# Contributing

PolicyPDEs is an early, unregistered Julia package. Start with docs/design.md to
understand the intended problem -> spatial discretization -> solve workflow.
The numerical solver and public API are not implemented yet.

Use Julia 1.10 or newer. Clone the repository, activate its root environment,
and run the package tests:

```julia
using Pkg
Pkg.activate(".")
Pkg.test()
```

For numerical contributions, include a small reproducible example and tests
against a known solution or independent invariant. Check nonlinear residuals
for backward solvers and conservation and nonnegativity for forward solvers.
Explain boundary conditions and mass-versus-density conventions explicitly.

Keep runtime dependencies minimal, add compatibility bounds, and avoid committing
Manifest.toml files. Public API changes should update the design documentation
and examples. Open a GitHub issue for design discussion or submit a pull request.
