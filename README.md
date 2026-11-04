# PolicyPDEs

[![Build Status](https://github.com/KaiJyunWang/PolicyPDEs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KaiJyunWang/PolicyPDEs.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia package under development for economic PDEs that are linear conditional on a policy.

**Status:** problem setup, fixed-control HJB coefficient/residual evaluation,
and linear boundary definitions/evaluation are implemented. Spatial
discretization and numerical solvers are not implemented yet.

## Installation

Requires Julia 1.10 or newer. Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/KaiJyunWang/PolicyPDEs.jl")
using PolicyPDEs
```

## Vector-valued problem setup

The unknown is `v: R^d -> R^m`, with `d` continuous coordinates and `m`
CTMC states. Only continuous coordinates belong in `domain`; `regimes`
determines the order of value components and coefficient columns.

Every HJB primitive takes **`(x, control)`**, returning all discrete components
at once. User-defined callable structs and ordinary functions are supported.

| Primitive | Output shape | Entry meaning |
|---|---|---|
| `discount` | length `m` | `rho[i]` |
| `flow` | length `m` | `u[i]` |
| `drift` | `d × m` | coordinate `k`, regime `i` |
| `diffusion` | `d × m` | coordinate volatility `sigma[k,i]` |
| `transition` | `m × m` | intensity from row `i` to column `j` |

The transition diagonal must be zero. This is an off-diagonal rate matrix,
with the generator diagonal derived from outgoing rates. `transition=nothing`
means zero switching. The volatility matrix's columns index regimes; each
continuous coordinate has an independent Brownian shock. There are no mixed
second derivatives. Shapes stay explicit even for `d=1` or `m=1`.

```julia
using PolicyPDEs

discount = (x,c) -> [0.05, 0.05]
flow = (x,c) -> log.(c)
drift = (x,c) -> reshape([0.1,0.2] .+ 0.03*x .- c, 1, 2)
diffusion = (x,c) -> zeros(1,2)
transition = (x,c) -> [0.0 0.02; 0.03 0.0]
domain = (; assets=(0.0,5.0))

# Boundary coefficient callbacks have a separate (x, regime) interface.
marginal = (x,i) -> inv([0.1,0.2][i] + 0.03*x)
boundary = (; assets=(
    lower=LinearInequalityBoundary(a=-1,b=0,c=(x,i)->-marginal(x,i)),
    upper=LinearEqualityBoundary(a=1,b=0,c=marginal),
))
problem = HJBProblem(;discount,drift,diffusion,flow,domain,boundary,
                     regimes=(1,2),transition)
```

The control is passed unchanged. In this example `c=[c_low,c_high]` holds a
control for each regime. Callbacks may instead define structured or shared
candidate controls. For row-wise policy improvement, coefficient column `i`,
discount/flow entry `i`, and transition row `i` depend on source control
`c_i`. All five primitives may depend on continuous state and control.

See the [callable-struct example](examples/two_income_setup.jl) and the
[two-coordinate example](examples/independent_diffusion_setup.jl).

## Evaluating all equations together

```julia
control = [1.0,1.0]
coefficients = hjb_coefficients(problem,1.0,control)
residuals = hjb_residual(problem,1.0,control,[1.0,3.0],zeros(1,2),zeros(1,2))
# approximately [-0.01,-0.21]
```

`hjb_coefficients` evaluates all five callbacks once and validates their shapes
and finite real entries, including nonnegative off-diagonal rates.
`hjb_residual` returns a length-`m` vector, using a length-`m` value vector
and `d × m` first-derivative and pure-second-derivative matrices.
Entry `[k,i]` refers to continuous coordinate `k` of value component `i`.

For fixed controls, component `i` is RHS minus LHS:

```text
u[i] - rho[i]*v[i]
  + sum(mu[k,i]*partial_k(v_i) + sigma[k,i]^2*partial_kk(v_i)/2 for k in 1:d)
  + sum(rates[i,j]*(v[j]-v[i]) for j != i).
```

The Bellman equation maximizes each complete residual over feasible source
controls, including control-dependent discounting. Switching preserves the
continuous state. The evaluation helpers do not differentiate, optimize,
assemble a grid, or solve a PDE.

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
# Spatial discretization and solve API are proposed, not implemented yet.
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
