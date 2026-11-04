# Design plan

## Problem -> spatial discretization -> solve

Problem setup describes economic primitives, a rectangular continuous domain,
and lower/upper face conditions. Spatial discretization will own grids, derivative
operators, boundary enforcement, and assembly. The solver will invoke user-supplied policy updates and own iteration,
initial guesses, pseudo-time steps, tolerances, linear solves, and diagnostics.
Only setup and pointwise equation/boundary evaluation are implemented so far.

## Rectangular geometry

`RectangularDomain(lower,upper)` accepts nonempty tuples/vectors of finite real
bounds with equal lengths and strict lower<upper ordering in every coordinate.
Scalars are accepted for a one-dimensional interval. Bounds are snapshotted into
tuples without converting numeric entries. `length(domain)` is d; membership
includes boundaries/corners. Coordinates are ordered by position rather than names.
Only continuous coordinates belong in this domain.

`RectangularBoundary(lower,upper)` stores a tuple of conditions on each side,
with one pair per coordinate. A single condition on each side is allowed in
one dimension. Each face holds LinearEqualityBoundary or LinearInequalityBoundary.
Condition k on the lower side belongs to x[k]=domain.lower[k], and similarly
for the upper side. HJBProblem requires both types and checks the face count
against the domain dimension, without evaluating user callbacks.

## Bulk HJB primitives

```julia
HJBProblem(discount,drift,diffusion,flow;control,domain,boundary,transition=nothing)
HJBProblem(;discount,drift,diffusion,flow,control,domain,boundary,transition=nothing)
```

The unknown is v: R^d -> R^m. All HJB primitives use (x,control), with no
discrete-state input. Discount/flow are length-m vectors or tuples;
drift/diffusion are d-by-m matrices, coordinate by value component.
Transition is an m-by-m matrix of off-diagonal rates, source row by destination
column, with zero diagonal. Generator diagonals equal minus outgoing rate sums.
Transition=nothing means zero switching.

No labels or component count are stored. Infer m from the nonempty discount
vector at evaluation, then validate the other outputs. Callbacks must keep a
fixed m across states/controls; the future discretizer will track this across
grid points. Even singleton dimensions remain explicit. Coefficient output
positions share one ordering.

The volatility entries enumerate independent coordinate Brownian motions.
Each sigma[k,i]^2/2 multiplies partial_kk(v_i); there are no mixed derivatives,
cross-coordinate covariance, or spatial resets upon switching. The opaque control
object passes unchanged to each callback. For row-wise policy improvement,
source-i entries/columns and transition row i depend on source control i.
Cross-component control dependence needs a different optimization contract.

`hjb_coefficients(problem,x,control)` calls the supplied HJB primitives once,
validating finite real entries, shapes, and nonnegative transition rates.
State x must have dimension d and lie in the closed rectangular domain.
`hjb_residual(problem,x,control,values,gradient,second_derivatives)` returns all
m residual components. Values are length m; derivatives are d-by-m matrices.

For fixed controls, component i is

    flow[i] - discount[i]*values[i]
      + sum(drift[k,i]*gradient[k,i]
            + diffusion[k,i]^2*second_derivatives[k,i]/2 for k in 1:d)
      + sum(transition[i,j]*(values[j]-values[i]) for j != i).

The Bellman maximum includes the entire residual, including control-dependent
discounting. These helpers do not differentiate, optimize, or solve a PDE.
CTMC switching preserves x.

## User-supplied control update

HJBProblem requires a callable `control(v,Dv,D2v,x)`. It is stored with its
concrete type and is not evaluated at construction. v is a nonempty vector/tuple;
Dv and D2v are d-by-m matrices of first and pure second derivatives. Mixed
partials are not supplied for the initial independent-coordinate diffusion class.
The callback must treat arguments as read-only and return an AbstractVector
of finite real controls, without a discrete-state argument. Length/layout is
model-defined; multiple controls may use a documented flattened ordering.
The returned vector is passed unchanged to the economic primitives.

`update_control(problem,x,v,Dv,D2v)` checks geometry/input dimensions and calls
the callback once. It validates the vector result without copying/conversion
or calling HJB coefficient functions. This avoids circular dependence on
control-dependent discounting when determining a policy. Scalar, tuple, matrix,
nonreal, or nonfinite outputs are rejected.

`hjb_residual(problem,x,v,Dv,D2v)` updates controls and delegates to the existing
explicit-control residual evaluator. This is the Bellman residual only if the
user actually supplies a feasible maximizer; validation does not certify that.
Row-wise maximization must account for the complete residual, including effects
on discount, volatility, and transition rates. The solver iteration engine and
numerical differentiation remain future work.

The log-consumption example supplies a callable struct using inverse marginal
value and enforcing endpoint consumption feasibility. The two-coordinate
example supplies a maximizer over consumption candidates 0.1:0.1:1.0, using
values, both derivatives, state-dependent coefficients, and CTMC gains.

## Componentwise linear face conditions

LinearEqualityBoundary(a,b,c) describes a[i]*partial_k(v_i)+b[i]*v_i=c[i].
LinearInequalityBoundary uses <=. Coefficients may be finite real scalars,
vectors/tuples, or functions/callable structs of the full continuous state x,
returning scalars/vectors/tuples. Scalars broadcast to m. No callback receives
a component label. Each component must have a nonzero a or b.

Constant coefficients are checked during condition construction without calling
user functions. Evaluated vectors must match m, inferred from values in residual
evaluation or supplied to boundary_coefficients as ncomponents. If ncomponents is
omitted, infer it from coefficient vectors, falling back to one for all scalars.

The derivative uses the increasing coordinate direction on both lower/upper
faces. Outward-normal lower-face conditions require negating a. Express >= by
negating a,b,c together. No sign normalization is performed.

`boundary_coefficients(condition,x;ncomponents=m)` returns component
vectors/tuples a,b,c. `boundary_residual(condition,values,derivatives;x)`
returns a[i]*derivatives[i]+b[i]*values[i]-c[i] for all components.
Equality residuals must be zero; inequality residuals must be nonpositive.

The problem overload
`boundary_residual(problem,values,gradient;x,axis,side)`
selects one face, verifies x is in the rectangle/on that face, and extracts
the corresponding coordinate gradient row. It never evaluates HJB callbacks.
At corners/edges, each incident face is evaluated separately. Consistency,
tolerances, active-set/complementarity treatment, and numerical enforcement
remain discretizer/solver responsibilities.

## Examples and policy iteration

The [two-income example](../examples/two_income_setup.jl) follows the HJB
structure of [Achdou et al. (2022), REStud 89(1), 45-86](https://benjaminmoll.com/wp-content/uploads/2019/07/HACT.pdf)
on assets [0,5]. Parameters are illustrative. Its lower marginal-value inequality
follows no borrowing and the log-utility consumption first-order condition.
The upper marginal-value equality is an explicit zero-saving truncation choice.
It is not a general replacement for state constraints.
[The two-coordinate example](../examples/independent_diffusion_setup.jl) uses a
rectangle with vector bounds and distinct conditions for each coordinate face.

With policy fixed, stack the values into a block linear system with spatial
and CTMC generator L. Define A=diag(rho(policy))-L and b=flow(policy), with
appropriate boundary contributions. The semi-implicit step is

    (A + I/dt)*v_next = b + v_current/dt.

At dt=Inf, remove the pseudo-time terms explicitly to obtain Howard iteration.
Start with fixed dt. Stop using the nonlinear residual with a policy greedy
for the updated values; small value changes alone can be misleading at small dt.

## Forward problem and milestones

The planned forward API will reuse the policy-conditioned generator with
consistent boundary handling. For a conservative backward generator L,
probability mass evolves under transpose(L). Densities with integration weights
W use inv(W)*transpose(L)*W. Use mass internally and offer density conversion.
Stationary distributions need normalization and nonuniqueness diagnostics.
Transient forward solves use physical time, distinct from HJB pseudo-time.

Initial implementation milestones remain:
- A stationary consumption-saving HJB with one continuous asset coordinate.
- Structured-grid upwinding, state constraints, and sparse policy evaluation.
- Finite-dt semi-implicit iteration and Howard iteration.
- Stationary/transient forward solves under a fixed policy.
- Known/manufactured solutions, residual and grid-refinement checks, and conservation.
- A documented end-to-end solve before stabilizing the public API.

Equilibrium loops, finite-horizon HJBs, adaptive grids, and correlated diffusions
are subsequent extensions.
