# PolicyPDEs

[![Build Status](https://github.com/KaiJyunWang/PolicyPDEs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KaiJyunWang/PolicyPDEs.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia package under development for economic PDEs that are linear conditional on a policy.

**Status:** early, unregistered package. Problem setup, fixed-control HJB
coefficient/residual evaluation, and linear boundary definitions/evaluation are
implemented; spatial discretization and numerical solvers are not implemented yet.

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

The initial equation class is **independent coordinate Brownian motions plus a
finite-state CTMC**. With `d` continuous coordinates and `m` regime labels, the
unknown is `v: R^d -> R^m`. For the source component `i`, a fixed control `c`
satisfies

```text
rho_i(x,c) * v_i(x) = u_i(x,c)
    + sum(mu_ki(x,c) * partial_k(v_i) for k in 1:d)
    + 0.5 * sum(sigma_ki(x,c)^2 * partial_kk(v_i) for k in 1:d)
    + sum(q_ij(x,c) * (v_j(x) - v_i(x)) for j != i).
```

The field names map to `rho=discount`, `u=flow`, `mu=drift`, `sigma=diffusion`,
and `q=transition`. Every primitive is a function of state and control (constant
parameters can be represented by callbacks that ignore those arguments). The
discrete label selects an output component, so only continuous coordinates
belong in `domain`. `regimes` determines the order of entries in `v(x)`.

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
`diffusion` supplies one volatility per continuous coordinate, as a tuple or
vector. The second-order coefficients are `sigma[k]^2 / 2`; there are no mixed
second derivatives. Scalar drift and volatility are also accepted in one
dimension. Matrix-valued diffusion is outside the initial supported equation class.

Inputs retain their types without copying or callback evaluation. Construction
checks for nonempty, unique regime labels. Evaluation helpers validate callback
results without changing their types. Numerical boundary enforcement and policy
optimization await the spatial discretization and solver interfaces.
See `?HJBProblem` and the [callable-struct example](examples/two_income_setup.jl).

## Evaluating the equation

```julia
coefficients = hjb_coefficients(problem, 1.0, 1.0, 1)
# v(x) has two components. Supply derivatives for component 1 only.
residual = hjb_residual(problem, 1.0, 1.0, [1.0, 3.0], 0.0, 0.0; regime=1)
# residual is approximately -0.01, using the two-income problem above.
```

The residual is RHS minus LHS for a supplied control. The Bellman equation
maximizes this entire residual over feasible controls, including the term
`-rho(x,c)*v_i(x)` when discounting depends on the control. Each source regime
uses its own control when evaluating coefficients and outgoing transition rates.
`hjb_residual` checks the value-vector length, derivative dimensions, and finite
nonnegative off-diagonal rates. It does not compute derivatives or optimize controls.

For multiple continuous coordinates, pass tuple/vector states, drift, volatility,
gradients, and pure second derivatives in domain-axis order. See the
[two-coordinate setup example](examples/independent_diffusion_setup.jl).

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
