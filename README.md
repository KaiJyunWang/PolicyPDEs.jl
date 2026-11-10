# PolicyPDEs

[![Build Status](https://github.com/KaiJyunWang/PolicyPDEs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KaiJyunWang/PolicyPDEs.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia package under development for economic PDEs that are linear conditional on a policy.

**Status:** rectangular problem setup, derivative-dependent coefficient callbacks, HJB residual
evaluation, linear face boundary evaluation, and finite-difference
grid/stencil preparation, sparse operator assembly, and Picard/Howard solves with
boundary enforcement are implemented. The forward API remains planned.

## Installation

Requires Julia 1.10 or newer:

```julia
using Pkg
Pkg.add(url = "https://github.com/KaiJyunWang/PolicyPDEs.jl")
using PolicyPDEs
```

## Rectangular domains and faces

Supply a lower and upper bound for each continuous coordinate:

```julia
domain = RectangularDomain(lower = [0.0, 0.5], upper = [5.0, 1.5])
boundary = RectangularBoundary(
    lower = (LinearInequalityBoundary(-1, 0, -2), LinearBoundary(1, 0, 0)),
    upper = (LinearBoundary(1, 0, 0), LinearBoundary(0, 1, 0)),
)
```

This describes `[0, 5] × [0.5, 1.5]`. Bounds must be finite and strictly ordered.
Bounds are stored as tuples, taking snapshots of input vectors.
`length(domain)` gives the continuous dimension; `x in domain` includes faces
and corners. Scalars may specify a one-dimensional interval.

`boundary.lower[k]` applies on `x[k] = domain.lower[k]`, and
`boundary.upper[k]` applies on `x[k] = domain.upper[k]`. There must be one
condition for each of the 2d faces. In one dimension, a single condition per
side is accepted without wrapping it in a tuple.

## Vector-valued HJB coefficients

The unknown is `v: R^d -> R^m`. Every coefficient callback takes
**`(x, v, Dv, D2v)`** and returns all value components together. There is no
`control` callback or control argument. `v` has length m; `Dv` and `D2v` have
shape d-by-m and contain first and pure second coordinate derivatives.

| Callback | Output shape | Meaning |
|---|---|---|
| `discount` | length m vector/tuple | discount rates |
| `flow` | length m vector/tuple | instantaneous payoffs |
| `drift` | d-by-m matrix | coordinate row, value-component column |
| `diffusion` | d-by-m matrix | independent-coordinate volatilities |
| `transition` | m-by-m matrix | source row, destination column |

Transition diagonals must be zero; off-diagonal rates are nonnegative.
`transition = nothing` means no switching. The diffusion term is
`sigma[k, i]^2 * D2v[k, i] / 2`, with no mixed derivatives. Shapes stay explicit
for one coordinate or one value component. Treat callback inputs as read-only.

For example, this illustrative nonlinear system has two value components:

```julia
discount(x, v, Dv, D2v) = [1.0, 1.0]
flow(x, v, Dv, D2v) = [1.0, 2.0] .+ 0.1 .* tanh.(v)
drift(x, v, Dv, D2v) = reshape(-0.1 .* Dv[1, :], 1, 2)
diffusion(x, v, Dv, D2v) = [0.2 0.3]
transition(x, v, Dv, D2v) = [0.0 0.02; 0.03 0.0]
domain = RectangularDomain(0.0, 5.0)
boundary = RectangularBoundary(
    lower = LinearBoundary(0, 1, [1.0, 2.0]),
    upper = LinearBoundary(0, 1, [1.0, 2.0]),
)
problem = HJBProblem(; discount, flow, drift, diffusion, transition, domain, boundary)
```

Callbacks encode the model directly. If an economic control is useful internally,
compute it inside a coefficient function or a shared model helper. The package
validates shapes, finiteness, and transition rates; it does not infer an optimizer
or certify economic optimality. See the self-contained
[two-income example](examples/solve_two_income.jl).

## Pointwise evaluation and boundary definitions

```julia
v = [1.0, 2.0]
Dv = zeros(1, 2)
D2v = zeros(1, 2)
coefficients = hjb_coefficients(problem, 1.0, v, Dv, D2v)
residual = hjb_residual(problem, 1.0, v, Dv, D2v)
```

Each supplied coefficient is called once with the same input objects.
The residual is HJB RHS minus LHS, including CTMC switching gains.
Pointwise evaluation does not select spatial upwinding. The nonzero component
count comes from `v`, and every output must agree with it.

`LinearBoundary(a, b, c)` defines `a[i] * partial_k(v_i) + b[i] * v_i = c[i]`;
`LinearInequalityBoundary` uses <= . Boundary coefficients are scalar broadcasts,
vectors/tuples, or functions of **x only**. Derivatives point toward increasing
coordinates at both faces. Negate all three coefficients to express >= .

`boundary_coefficients(condition, x; ncomponents = m)` evaluates face coefficients.
`boundary_residual(condition, v, Dv_face; x)` checks a face from supplied values
and its coordinate derivative vector. The problem overload is
`boundary_residual(problem, v, Dv; x, axis, side)`. At corners, pointwise evaluation
can inspect every face separately; the solver's row priority is described below.

For an equality with value- or derivative-dependent coefficients, use
`NonlinearBoundary(a, b, c)`. Constants still broadcast; callbacks take
`(x, v, Dv, D2v)` and return scalars or length-m vectors/tuples:

```julia
condition = NonlinearBoundary(
    a = (x, v, Dv, D2v) -> 1 .+ 0.1 .* v .^ 2,
    b = 1,
    c = (x, v, Dv, D2v) -> 2 .+ 0.05 .* Dv[1, :] .+ 0.01 .* D2v[1, :],
)
coefficients = boundary_coefficients(condition, 1.0, v, Dv, D2v)
residual = boundary_residual(condition, v, Dv, D2v; x = 1.0, axis = 1)
```

Unlike `LinearBoundary`, these coefficients use the current values and full
derivative matrices. The solver evaluates them from unmodified finite differences,
freezes them for each linear row replacement, and refreshes them before checking
the actual nonlinear boundary residual. For the problem helper, pass second
derivatives as `boundary_residual(problem, v, Dv, D2v; x, axis, side)`.
Arbitrary nonlinear boundary iterations need not converge.

## Spatial discretization interface

Put the grid specification inside `FiniteDifference(...)`:

```julia
# One continuous coordinate: 101 equidistant points, including endpoints.
discrete = discretize(problem, FiniteDifference((101,)))

# The same problem with a nonuniform user-supplied coordinate grid.
discrete = discretize(problem, FiniteDifference(([0.0, 0.01, 0.1, 1.0, 5.0],)))
```

For two continuous coordinates, use `FiniteDifference((101, 51))` for
equidistant discretization, or `FiniteDifference((xgrid, ygrid))` for supplied
one-dimensional grids. `FiniteDifference(xgrid, ygrid)` is also accepted.
The tuple length must equal the continuous dimension. Every coordinate needs
at least three points, and counts include endpoints. Supplied grids may be
nonuniform; they must be finite, strictly increasing, and have endpoints equal
to the corresponding problem bounds. Counts and coordinate grids cannot be mixed.

`discretize(problem, method)` creates a `DiscretizedHJBProblem` with a
`TensorGrid`, the original problem/method, and local stencil data. Grid
coordinates and steps are stored per axis, not as a materialized tensor of states.
`grid.coordinates[k]` contains physical coordinates and `grid.steps[k][j]`
the distance from node j to j+1. Linear iteration varies coordinate 1 fastest.
It returns scalar states in 1D and coordinate tuples otherwise. Value components
remain separate from the spatial grid.

The standalone `equidistant(domain, npoints)` helper is still available if
coordinate ranges are needed separately. They can be supplied through
`FiniteDifference(grid.coordinates)`.

```julia
stencil = fd_stencil(0.1; derivative = 2) # uniform three-point stencil
nonuniform = fd_stencil((0.1, 0.3); derivative = 2) # left and right distances
```

Access `discrete.stencils[k][j]` for coordinate k and node j, with
`.forward`, `.backward`, `.central`, and `.second` fields.
Weights use each node's actual adjacent spacings. Forward/backward derivatives
use two points; centered first and second derivatives differentiate the local
quadratic interpolant. Uniform centered stencils are second-order accurate;
arbitrary nonuniform second-derivative stencils need not have that convergence rate.

For the backward HJB generator, positive drift uses forward differences and
negative drift uses backward differences. These and the nonuniform second
derivative follow the [Achdou numerical appendix](https://benjaminmoll.com/wp-content/uploads/2020/02/HACT_Numerical_Appendix.pdf).
Offsets are relative coordinate-node indices.

Unavailable face stencils are `nothing`: no backward stencil at the lower
endpoint, no forward stencil at the upper endpoint, and no centered stencil
at either endpoint. Boundary conditions are not silently replaced by extrapolation.
No user callback is evaluated during spatial preparation. The solver constructs
sparse derivative operators with Kronecker products, multiplies coefficient
diagonals on the left, and adds the CTMC generator.

## Solve workflow

```julia
solution = solve(discrete; dt = 100.0, initial_value, show_trace = true);
solution.converged
solution.values       # node-by-value-component matrix
solution.coefficients # final nodal coefficient fields
solution.residual_norm
```

The solver always uses Picard-type iteration, with `dt = Inf` by default.
Finite pseudo-time steps regularize coefficient evaluation; dt = Inf gives full
frozen-coefficient iteration. Linear evaluation uses LinearSolve (sparse UMFPACK
by default). Supply `initial_value` as a matrix or `x -> vector` callback;
coefficients must be defined at the initial values and derivatives. Convergence
checks the stationary residual after a fresh coefficient update and the
coefficient change (`coefficient_tol`). Inspect `converged` and `retcode`.
Start with the [step-by-step examples guide](examples/README.md) and
[runnable consumption-saving example](examples/solve_two_income.jl).

Boundary equations replace HJB rows without pseudo-time terms. For an inequality
residual B <= 0 and HJB residual R = LHS - RHS, the active set solves
`max(sign(q) * R, B) = 0`, where q is the boundary row's own-node coefficient.
Derivative inequalities use inward first-order differences; derivative equalities
use one-sided quadratic formulas. At corners, equality Dirichlet conditions take
priority, followed by value inequalities and coordinate order. Only the selected
face is imposed; conflicting Dirichlet values are rejected.

For a drift-constrained endpoint use `StateConstraintBoundary(derivative)`.
The state-only scalar/vector callback supplies the derivative giving zero
normal drift, only for the missing outward difference. The inward difference
and the endpoint HJB are retained. No derivative equation is imposed on the
adjacent interior node. Normal diffusion must vanish. The two-income example
uses `StateConstraintBoundary(a -> 1 ./ resources(a))` at both asset endpoints.

The initial solver uses Float64 and independent coordinate diffusion. Coefficient
updates evaluate forward/backward derivatives first: (+, -) compares Hamiltonians,
(+, +) selects forward, (-, -) selects backward, and (-, +) requires a zero-drift
coefficient evaluation. The latter is found by bisection through the supplied continuous coefficient
callbacks, with its drift set exactly to zero in the operator. Coordinate choices
must stabilize in multiple dimensions. Derivative dependence must be local to each value component's derivative column.
Interior upwinding does not guarantee monotonicity of arbitrary endpoint closures.
The forward API remains planned and needs conservative boundary treatment.

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
