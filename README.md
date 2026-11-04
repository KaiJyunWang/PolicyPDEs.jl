# PolicyPDEs

[![Build Status](https://github.com/KaiJyunWang/PolicyPDEs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KaiJyunWang/PolicyPDEs.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia package under development for economic PDEs that are linear conditional on a policy.

**Status:** rectangular problem setup, fixed-control HJB evaluation, and linear
face boundary evaluation are implemented. Spatial discretization and numerical
solvers are not implemented yet.

## Installation

Requires Julia 1.10 or newer:

```julia
using Pkg
Pkg.add(url="https://github.com/KaiJyunWang/PolicyPDEs.jl")
using PolicyPDEs
```

## Rectangular domains and faces

Supply a lower and upper bound for each continuous coordinate:

```julia
domain = RectangularDomain(lower=[0.0,0.5], upper=[5.0,1.5])
boundary = RectangularBoundary(
    lower=(LinearInequalityBoundary(-1,0,-2), LinearEqualityBoundary(1,0,0)),
    upper=(LinearEqualityBoundary(1,0,0), LinearEqualityBoundary(0,1,0)),
)
```

This describes `[0,5] × [0.5,1.5]`. Bounds must be finite and strictly ordered.
Bounds are stored as tuples, taking snapshots of input vectors.
`length(domain)` gives the continuous dimension; `x in domain` includes faces
and corners. Scalars may specify a one-dimensional interval.

`boundary.lower[k]` applies on `x[k]=domain.lower[k]`, and
`boundary.upper[k]` applies on `x[k]=domain.upper[k]`. There must be one
condition for each of the 2d faces. In one dimension, a single condition per
side is accepted without wrapping it in a tuple.

## Vector-valued HJB primitives

The unknown is `v: R^d -> R^m`. CTMC states index output components.
Every primitive is a callable struct/function of **`(x, control)`**, returning
all components at once:

| Primitive | Output shape | Meaning |
|---|---|---|
| `discount` | length m vector/tuple | discount rates |
| `flow` | length m vector/tuple | instantaneous payoffs |
| `drift` | d × m matrix | coordinate row, component column |
| `diffusion` | d × m matrix | independent-coordinate volatilities |
| `transition` | m × m matrix | source row, destination column |

The nonzero component count m is inferred from discount. All other shapes must
agree; callbacks must keep the same m across states/controls. Transition
diagonals are zero; the generator diagonal is derived from outgoing rates.
`transition=nothing` means no switching. Independent coordinate diffusion
uses sigma[k,i]^2/2 on the pure second derivative, with no mixed derivatives.
Shapes remain explicit when d=1 or m=1.

```julia
discount = (x,c)->[0.05,0.05]
flow = (x,c)->log.(c)
drift = (x,c)->reshape([0.1,0.2] .+ 0.03*x .- c,1,2)
diffusion = (x,c)->zeros(1,2)
transition = (x,c)->[0.0 0.02;0.03 0.0]
domain = RectangularDomain(lower=[0.0],upper=[5.0])
marginal = x->inv.([0.1,0.2] .+ 0.03*x)
boundary = RectangularBoundary(
    lower=LinearInequalityBoundary(a=-1,b=0,c=x->-marginal(x)),
    upper=LinearEqualityBoundary(a=1,b=0,c=marginal),
)
problem = HJBProblem(;discount,drift,diffusion,flow,domain,boundary,transition)
```

Controls pass unchanged. Here a two-entry vector holds one control per component.
Source-i coefficient entries/columns and transition row i should depend on
source control i for row-wise policy improvement. All five primitives may
depend on continuous state and control. See the
[callable-struct example](examples/two_income_setup.jl) and
[two-coordinate example](examples/independent_diffusion_setup.jl).

## Evaluating the equations and boundaries

```julia
coefficients = hjb_coefficients(problem,1.0,[1.0,1.0])
residuals = hjb_residual(problem,1.0,[1.0,1.0],[1.0,3.0],zeros(1,2),zeros(1,2))
# approximately [-0.01,-0.21]

boundary_residual(problem,[0.0,0.0],reshape([10.0,5.0],1,2);
                  x=0.0,axis=1,side=:lower) # [0.0,0.0]
```

Each HJB primitive is called once. Coefficient evaluation checks shapes and finite
real values, including nonnegative off-diagonal rates. State points must lie
in the closed rectangle. Residual inputs are a length-m value vector/tuple and
d-by-m matrices of first and pure second coordinate derivatives.

`LinearEqualityBoundary(a,b,c)` defines
`a[i]*partial_k(v_i)+b[i]*v_i=c[i]`, and `LinearInequalityBoundary` uses <=.
Each coefficient may be a finite real scalar (broadcast to all components),
a vector/tuple, or a callable `coefficient(x)` returning a scalar/vector/tuple.
No boundary callback needs a discrete input. At least one of a[i],b[i] must be
nonzero. The derivative is in the increasing coordinate direction at both
faces; negate a on lower faces when expressing an outward-normal condition.
For >=, negate all three coefficients.

`boundary_coefficients(condition,x;ncomponents=m)` evaluates one face.
`boundary_residual(condition,values,derivatives;x)` evaluates that face from
vectors of values and its coordinate derivatives. The problem overload above
selects the face, extracts the gradient row, and checks that x lies on the face.
Residuals must be zero for equalities and nonpositive for inequalities.
At corners, evaluate each incident face separately. Coefficient validation does
not establish corner compatibility or enforce the condition numerically.

The example's lower inequality expresses the log-utility no-borrowing marginal
condition. Its upper equality is an illustrative zero-saving truncation choice.
The fixed-control HJB residual is RHS minus LHS, including switching gains.
The Bellman equation maximizes each complete residual over feasible controls,
including the discount term when discounting depends on control. Evaluation
does not differentiate, optimize, discretize, or solve a PDE.

## Planned workflow

```julia
# Spatial discretization and solving are proposed APIs.
discrete = discretize(problem,grid,UpwindFiniteDifference())
solution = solve(discrete,SemiImplicit(dt=Inf);initial_value)
forward_problem = forward(discrete,solution.policy)
distribution = solve(forward_problem,StationaryDistribution())
```

Finite pseudo-time steps regularize policy evaluation; dt=Inf gives Howard
iteration. The forward API will reuse the policy-conditioned spatial generator.

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
