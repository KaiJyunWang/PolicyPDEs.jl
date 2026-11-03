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

An `HJBProblem` takes four separate components: `discount`, `drift`, `diffusion`,
and `flow`. Each is a callable user-defined struct or a function taking the
continuous state, control, and regime label `(x, a, i)`.

```julia
using PolicyPDEs

discount = (x, c, i) -> 0.05
drift = (x, c, i) -> (i == 1 ? 0.1 : 0.2) + 0.03 * x - c
diffusion = (x, c, i) -> 0.0
flow = (x, c, i) -> log(c)
transition = (x, c, i, j) -> i == 1 ? 0.02 : 0.03
domain = (; assets = (0.0, 100.0))
boundary = (; assets = (lower = :state_constraint, upper = :state_constraint))
problem = HJBProblem(; discount, drift, diffusion, flow, domain, boundary,
                     regimes = (1, 2), transition)
```

`transition(x, a, i, j)` supplies the nonnegative intensity from regime `i` to
regime `j`, for `i != j`. The coupling is
`sum(q_ij * (v_j - v_i) for j != i)`, as in finite-income-state HJB systems.
It may depend on the continuous state and source-regime control. By default,
`regimes = (1,)` describes a single PDE; `transition = nothing` means no switching.
`diffusion` supplies volatility sigma: the second-order operator uses
`sigma * sigma' / 2`, not sigma itself as a variance coefficient.

Inputs retain their types without copying or callback evaluation. Construction
checks for nonempty, unique regime labels. Boundary descriptors above illustrate
storage; numerical boundary treatment, return-shape and rate validation, and
policy optimization await the spatial discretization and solver interfaces.
See `?HJBProblem` and the [callable-struct example](examples/two_income_setup.jl).

## Planned workflow

Set up the economic problem, discretize space, then solve:

```julia
# Proposed API; not implemented yet.
problem = HJBProblem(; discount, drift, diffusion, flow, domain, boundary,
                     regimes = (1, 2), transition)
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
