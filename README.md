# PolicyPDEs

[![Build Status](https://github.com/KaiJyunWang/PolicyPDEs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KaiJyunWang/PolicyPDEs.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia package under development for economic PDEs that are linear conditional on a policy.

**Status:** early, unregistered package. `HJBProblem` construction is implemented;
spatial discretization and numerical solvers are not implemented yet.

## Installation

Requires Julia 1.10 or newer. Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/KaiJyunWang/PolicyPDEs.jl")
using PolicyPDEs
```

## Problem setup

An `HJBProblem` stores a user-defined model, a state domain, and explicit boundary
data. It preserves their types without evaluating the model or copying inputs.

```julia
using PolicyPDEs

model = (; discount = 0.05, interest = 0.03, income = 1.0)
domain = (; assets = (0.0, 100.0))
boundary = (; assets = (lower = :state_constraint, upper = :state_constraint))
problem = HJBProblem(model; domain, boundary)
```

The descriptors above illustrate storage. A discretizer will define how to
interpret model primitives, domains, and boundary conditions; this constructor
does not yet enforce such an interface or check mathematical well-posedness.
See `?HJBProblem` for the constructor contract.

## Planned workflow

Set up the economic problem, discretize space, then solve:

```julia
# Proposed API; not implemented yet.
problem = HJBProblem(model; domain, boundary)
discrete = discretize(problem, grid, UpwindFiniteDifference())
solution = solve(discrete, SemiImplicit(dt=Inf); initial_value)

forward_problem = forward(discrete, solution.policy)
distribution = solve(forward_problem, StationaryDistribution())
```

The backward solver will use semi-implicit pseudo-time iteration. Finite `dt`
regularizes the policy-conditioned linear solve; `dt = Inf` gives Howard iteration.
The forward API will support stationary distributions and transient evolution
under a fixed policy, sharing the backward problem's spatial discretization.

See [the design plan](docs/design.md), [contributing](CONTRIBUTING.md), and
[the registration checklist](docs/registration.md).

## Development

```sh
git clone https://github.com/KaiJyunWang/PolicyPDEs.jl.git
cd PolicyPDEs.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Generated with [PkgTemplates.jl](https://github.com/JuliaCI/PkgTemplates.jl).
Licensed under the [MIT license](LICENSE).
