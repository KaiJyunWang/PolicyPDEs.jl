# PolicyPDEs

[![Build Status](https://github.com/KaiJyunWang/PolicyPDEs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KaiJyunWang/PolicyPDEs.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia package under development for economic PDEs that are linear conditional on a policy.

**Status:** early, unregistered package. `HJBProblem` construction and linear
boundary definitions/evaluation are implemented; spatial discretization and
numerical solvers are not implemented yet.

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
domain = (; assets = (0.0, 5.0))
marginal = (x, i) -> inv((i == 1 ? 0.1 : 0.2) + 0.03*x)
boundary = (; assets = (
    lower = LinearInequalityBoundary(a=-1, b=0, c=(x, i) -> -marginal(x, i)),
    upper = LinearEqualityBoundary(a=1, b=0, c=marginal),
))
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
checks for nonempty, unique regime labels. Numerical boundary enforcement,
component return-shape and transition-rate validation, and
policy optimization await the spatial discretization and solver interfaces.
See `?HJBProblem` and the [callable-struct example](examples/two_income_setup.jl).

## Linear boundary conditions

Attach conditions to each named domain axis using `lower` and `upper`:

```julia
boundary = (; assets = (
    lower = LinearEqualityBoundary(a=0, b=1, c=0),  # v = 0
    upper = LinearInequalityBoundary(a=1, b=0, c=2), # v_x <= 2
))
```

`LinearEqualityBoundary` represents `a*v_x + b*v = c`;
`LinearInequalityBoundary` represents `a*v_x + b*v <= c`. Coefficients can be
finite real constants or callable structs/functions taking `(x, regime)`.
The derivative is in the **increasing coordinate direction at both endpoints**.
For `>=`, negate all three coefficients. A zero `a` gives a value condition;
a zero `b` gives a derivative condition. Both cannot be zero.

```julia
condition = LinearInequalityBoundary(a=-1, b=0, c=-2) # v_x >= 2
boundary_coefficients(condition, 0.0, 1) # (a=-1, b=0, c=-2)
boundary_residual(condition, 0.0, 3.0; x=0.0, regime=1) # -1.0
```

A residual must be zero for an equality and nonpositive for an inequality.
Constant coefficients are checked at construction; callable coefficients are
checked when evaluated. These helpers evaluate the mathematical condition;
they do not yet enforce it numerically or decide when an inequality binds.

For the log-utility example, the lower condition is
`v_x >= 1 / (income_i + r*x)`, corresponding to no borrowing under the interior
consumption first-order condition. The upper equality imposes that marginal
value at the truncation point `x=5`; it is an illustrative zero-saving boundary
choice, not a general equivalence between state constraints and equalities.

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
