# Package design

The public workflow is problem setup -> spatial discretization -> solve.
The initial solver handles stationary vector-valued PDEs on rectangles,
independent coordinate Brownian motion, and finite CTMC switching.

## Direct coefficient contract

```julia
HJBProblem(; discount, drift, diffusion, flow, transition = nothing, domain, boundary)
```

Every supplied coefficient is a callable function/struct of `(x, v, Dv, D2v)`.
There is no control callback or intermediate control vector in the API.
For d continuous coordinates and m value components, v has length m;
Dv and D2v are d-by-m first/pure-second coordinate derivatives. x is a scalar
in 1D and a tuple/vector otherwise. Mixed second derivatives are unsupported.

Discount and flow return length-m vectors/tuples. Drift and diffusion return
d-by-m matrices. Transition returns m-by-m off-diagonal intensities with zero
diagonal, source row/destination column. Transition = nothing omits switching.
The component count is taken from v, and all outputs must agree with it.
Inputs are read-only. Construction checks geometry without calling coefficients.

Component i's equation is

    discount[i]*v[i] = flow[i]
        + sum(drift[k, i]*Dv[k, i] + diffusion[k, i]^2*D2v[k, i]/2 for k)
        + sum(transition[i, j]*(v[j]-v[i]) for j != i).

Coefficients can depend on all values, including switching gains. Their
**derivative dependence must be local to the corresponding component column**,
so directional evaluations can be combined across components. An economic
optimizer or closed-form consumption rule can be an internal model helper;
it is not registered as part of HJBProblem. The package cannot certify economic
optimality, feasibility, or a particular Bellman interpretation.

`hjb_coefficients(problem, x, v, Dv, D2v)` calls each supplied coefficient once,
validates state bounds, input/output shapes, finite real entries and CTMC rates,
and returns the original result arrays in a named tuple.
`hjb_residual(problem, x, v, Dv, D2v)` returns RHS minus LHS. These pointwise helpers
neither discretize nor choose an upwind direction.

## Domain and boundary conditions

RectangularDomain(lower, upper) snapshots finite, strictly ordered bounds.
RectangularBoundary(lower = ..., upper = ...) stores one condition per coordinate
face. LinearBoundary(a, b, c) means a*partial_k(v)+b*v = c; an inequality
uses <= . Each coefficient can be scalar, vector/tuple, or callable with x only.
There is no change to this state-only boundary coefficient contract.
Coordinate derivatives point toward increasing coordinates on both faces.

NonlinearBoundary(a, b, c) defines an equality of the same form with coefficients
depending on (x, v, Dv, D2v). Each coefficient can also be a constant scalar or
component vector/tuple. Full derivative matrices have the same d-by-m layout as
HJB coefficients. Nonlinear boundary callbacks see unmodified centered/one-sided
finite differences, avoiding circular evaluation of the boundary equation itself.
Evaluated coefficients are frozen for one linear solve and refreshed at new values.
Convergence uses the actual refreshed boundary residual, not the frozen equation.
Nonlinear Dirichlet conditions share equality priority and corner conflict checks.
No general convergence guarantee is provided for nonlinear boundary maps.

Equalities replace HJB rows. For B = a*Dv+b*v-c <= 0, let R be HJB LHS-RHS
and q the own-node coefficient of the boundary row. The solver uses
max(sign(q)*R, B) = 0. A zero q is rejected. Inequality derivatives use inward
first-order differences; equality derivatives use one-sided quadratic formulas.
At corners choose equality Dirichlet first, then value inequalities, then
coordinate order. Only the selected condition is imposed per component;
conflicting Dirichlet values are rejected.

StateConstraintBoundary(derivative) has different semantics from a linear
derivative equation: it retains the endpoint HJB and permits inward drift.
Its scalar/vector or state-only callback supplies a zero-normal-drift derivative
only for the unavailable outward difference (backward at lower, forward at upper).
The available inward difference is untouched, including the backward difference
at the second grid point. The solver validates zero drift for this supplied
derivative and inward selected drift. Normal diffusion must be zero at the face;
this condition does not model a reflecting diffusion. Every state-constraint
face is enforced at corners; linear row conditions retain their existing priority.

## Finite differences and drift-sign selection

FiniteDifference((n1, n2, ...)) requests equidistant endpoint-inclusive grids.
FiniteDifference((xgrid, ygrid, ...)) accepts strictly increasing coordinate grids
matching domain endpoints. Each axis needs at least three nodes; explicit grids
may be nonuniform. Grid preparation evaluates no coefficient callbacks.
TensorGrid stores coordinate axes rather than materializing tensor states.
Coordinate 1 varies fastest, and values have shape N-by-m.

For adjacent distances l, r, the interior stencils are

    forward: (-1/r, 1/r)
    backward: (-1/l, 1/l)
    central: (-r/(l*(l+r)), (r-l)/(l*r), l/(r*(l+r)))
    second: (2/(l*(l+r)), -2/(l*r), 2/(r*(l+r))).

Uniform centered stencils are second order. Arbitrary nonuniform second
stencils need not have second-order convergence. Endpoint PDE closures use
inward first differences and one-sided quadratic second derivatives; they do
not guarantee generator monotonicity on inactive inequality faces.

For each coordinate/component, first evaluate coefficients using forward and
backward derivatives. For their respective drift signs:

| Forward | Backward | Choice |
|---|---|---|
| Positive | Negative | Compare Hamiltonians |
| Positive | Positive | Forward |
| Negative | Negative | Backward |
| Negative | Positive | Zero drift |

Zero endpoints are admissible; Hamiltonian ties choose forward. In the last
case, bisection in the derivative finds consistent zero-drift coefficients.
The map must be continuous and possess a root. A validated near-zero drift is
set exactly to zero in the frozen operator. Failure to find a root is an error.
Coordinate choices are repeated until stable in multiple dimensions; failure to
stabilize is also an error. No control vector is inferred or interpolated.

## Sparse assembly and iteration

spatial_operators lifts 1D sparse derivative matrices with Kronecker products.
assemble_operator(discrete, values) evaluates and freezes upwind coefficients,
then left-multiplies derivative operators by drift and variance diagonals.
CTMC blocks are assembled with sparse Kronecker products. The raw system is
A = diag(discount)-generator and b = flow, with component-block stacking under vec.
The returned named tuple exposes coefficients, generator, raw_A/raw_b, A/b,
and boundary data. It contains no controls and performs no solve.

```julia
solution = solve(discrete; dt = 100.0, initial_value, show_trace = true);
```

`solve` always uses Picard-type iteration, with the keyword `dt = Inf` by default.
Each step freezes evaluated coefficients and solves
(A+I/dt)*v_next = b+v_current/dt on HJB rows through LinearSolve.
Boundary rows receive no pseudo-time shift. dt = Inf removes pseudo-time terms,
giving full frozen-coefficient evaluation (Howard-type iteration).
The default linear algorithm is sparse UMFPACK; linear_algorithm can override it.

Convergence requires the stationary residual after a fresh coefficient update
and a sufficiently small coefficient change. coefficient_tol replaces the old
policy_tol option. Small value changes alone are insufficient. show_trace defaults
to false and prints only aligned iteration/residual columns when enabled.
MaxIters returns a nonconverged result; linear/active-set failures throw errors.

HJBSolution exposes values, coefficients, coefficient_change, convergence status,
residual/boundary residual, residual history, and the final linear system.
Nodal discount/flow have shape N-by-m; drift/diffusion d-by-N-by-m;
transition N-by-m-by-m. There is no solution.controls or update_control helper.
The current solver uses Float64. Arbitrary coefficient iterations need not converge.

The self-contained example is [two income states](../examples/solve_two_income.jl).

## Forward problem and later extensions

The forward API remains planned. A conservative backward generator L propagates
probability mass under transpose(L). Densities with integration weights W use
inv(W)*transpose(L)*W. Stationary distributions need normalization and
nonuniqueness diagnostics. Endpoint closures need conservative boundary treatment
before the frozen HJB operator can be reused for this purpose.
Finite-horizon HJBs, correlated diffusion, adaptive grids, and equilibrium loops
are subsequent extensions. Julia 1.13.2 is the development target; compat retains
Julia 1.10 pending CI verification.
