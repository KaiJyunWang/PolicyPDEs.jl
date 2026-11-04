# Design plan

## Implemented: bulk HJB problem API

```julia
HJBProblem(discount,drift,diffusion,flow;domain,boundary,regimes=(1,),transition=nothing)
HJBProblem(;discount,drift,diffusion,flow,domain,boundary,regimes=(1,),transition=nothing)
```

The problem describes `v: R^d -> R^m`. The continuous state x has d coordinates;
m CTMC states index output components, not domain/grid axes. Regime metadata is
a nonempty tuple/vector of unique labels. Domain-axis order determines continuous
rows; regime order determines vector entries, matrix columns, and switching
source/destination positions.

All five HJB primitives use `(x,control)`, with no discrete-state input:

- Discount and flow return length-m vectors (tuples also accepted).
- Drift and diffusion return d-by-m matrices, coordinate by regime.
- Transition returns an m-by-m matrix of off-diagonal rates, source by destination.
  Diagonal entries must be zero; generator diagonals are minus outgoing sums.
  A supplied callback must return a matrix; use `transition=nothing` to omit it.

The diagonal-diffusion convention is one independent coordinate Brownian motion
per continuous coordinate. The volatility matrix enumerates regimes by columns,
not shock loading directions. Entry sigma[k,i] contributes sigma[k,i]^2/2 to
partial_kk(v_i). No mixed derivatives, correlated coordinate innovations, or
spatial resets upon CTMC switches are included initially. Output matrix shapes
remain explicit for d=1 or m=1; no automatic squeezing/broadcasting is applied.

The opaque control object is passed unchanged to all callbacks. A vector may
hold one control per regime, a matrix may hold several controls per regime,
or callbacks may define a structured/shared candidate control. For row-wise
policy improvement, source-i entries and transition row i depend on source
control c_i only. Cross-regime control dependence would require a different
optimization interface and is outside the initial row-wise solver contract.

Construction preserves concrete callback types without evaluation/copying and
checks regime metadata. Domain/boundary data are required. Mutating shared
inputs changes the problem; regime membership and order must remain fixed.
Boundary coefficient callbacks remain a separate `(x,regime)` interface.

## Coefficients and fixed-control residuals

`hjb_coefficients(problem,x,control)` calls all supplied HJB primitives once,
returning discount, drift, diffusion, flow, and transition with their original
types/references. It checks shapes, finite real outputs, nonnegative
off-diagonal rates, and zero rate diagonals. A named-tuple domain's axis count
must match x, represented as a scalar in one dimension or tuple/vector.

`hjb_residual(problem,x,control,values,gradient,second_derivatives)` returns
all m residual components together. Values is a length-m vector/tuple;
derivatives are d-by-m matrices of first and pure second partial derivatives.
The residual for source i is

    flow[i] - discount[i]*values[i]
      + sum(drift[k,i]*gradient[k,i]
            + diffusion[k,i]^2*second_derivatives[k,i]/2 for k in 1:d)
      + sum(transition[i,j]*(values[j]-values[i]) for j != i).

Each callback is evaluated once for the full control object. There are no
regime-specific callback calls. CTMC switching keeps x fixed; the rate matrix
contributes exactly Q*v when the generator diagonal is derived as specified.
No transition callback means zero switching.

These are mathematical evaluation helpers, without numerical differentiation,
policy optimization, spatial assembly, or PDE solving. The Bellman equation
maximizes the entire residual, including discounting when it depends on control.
The fixed-policy system is linear in values and derivatives.

## Linear boundary definitions

The public boundary definitions are:

```julia
LinearEqualityBoundary(a, b, c)    # a*v_x + b*v = c
LinearInequalityBoundary(a, b, c)  # a*v_x + b*v <= c
```

Both also accept the keywords `a`, `b`, and `c`. The coefficients are finite
real constants or user-defined callables `coefficient(x, regime)`. Their types
are preserved. Constants are validated without evaluating user callbacks;
evaluated coefficients must be finite real scalars with at least one of a and
b nonzero. Coefficients are independent of the control and value function, so
the condition remains linear in the unknown values and derivatives.

Use the same axis names as the domain, with `lower` and `upper` entries:

```julia
domain = (; assets = (0.0, 5.0))
boundary = (; assets = (
    lower = LinearInequalityBoundary(a=-1, b=0, c=-2), # v_x >= 2
    upper = LinearEqualityBoundary(a=1, b=1, c=3),    # v_x + v = 3
))
```

The derivative is always with respect to the increasing axis coordinate,
including at the lower boundary. Outward-normal conditions require the user
to change the derivative coefficient's sign at a lower endpoint. An inequality
uses <=; a >= condition is expressed by negating all three coefficients. No
automatic sign normalization is performed.

`boundary_coefficients(condition, x, regime)` returns the validated named tuple
`(; a, b, c)`. `boundary_residual(condition, value, derivative; x, regime=1)`
returns a*derivative + b*value - c. Equality residuals must be zero and inequality
residuals nonpositive; tolerance selection belongs to the numerical solver.

The two-income example uses [0,5], a lower inequality v_x >= u'(income_i + r*x)
for log utility, and an illustrative upper equality v_x = u'(income_i + r*x).
The lower condition follows from the interior consumption first-order condition
and no borrowing; the upper equality is an explicit zero-saving truncation
choice. It is not a general replacement for state constraints. Boundary
enforcement and the active-set/complementarity treatment needed for inequalities
remain future numerical work. Merely storing an inequality does not impose an
equality where it is slack or establish a well-posed boundary value problem.

## CTMC examples and policy evaluation

[Achdou et al. (2022), REStud 89(1), 45-86](https://benjaminmoll.com/wp-content/uploads/2019/07/HACT.pdf)
uses the two-state terms lambda_1*(v_2-v_1) and lambda_2*(v_1-v_2).
The off-diagonal rate callback returns `[0 lambda_1; lambda_2 0]`.
[The callable-struct setup](../examples/two_income_setup.jl) uses assets on [0,5]
and a two-entry consumption vector. Parameters and upper boundary are illustrative,
not a replication calibration.
[The independent-diffusion setup](../examples/independent_diffusion_setup.jl)
has d=2 and m=2 with state/control-dependent coefficients and rates.

With policy fixed, stack the values into a block linear system. The full
generator contains spatial and CTMC terms. For state/control-dependent
discounting, A is diag(rho(policy)) minus that generator. The semi-implicit
step is `(A+I/dt)*v_next=flow(policy)+v_current/dt`. The forward distribution
problem and equilibrium feedback remain separate planned APIs.

## Public workflow

**Set up problem -> spatial discretization -> solve.**

1. A problem describes economic primitives, admissible controls, domain, and boundary conditions.
2. Spatial discretization supplies grids, derivative operators, boundary treatment,
   and policy-conditioned system assembly. A discretized HJB is still nonlinear.
3. The solver owns initialization, policy iteration, linear solves, tolerances,
   diagnostics, and termination. Results contain values, policies, and convergence history.

The core should allow users to supply policy updates and linear system assembly.
Convenience interfaces for economic primitives can be built above that core.
Policy selection and upwind stencil selection must remain consistent.

## Backward problem

For a constant discount rate rho independent of control, the discretized HJB is

    rho * v = max_a { r(a) + L(a) * v }.

At iteration n, improve the policy using v_n and solve

    [(rho + 1/dt) * I - L(a_n)] * v_next = r(a_n) + v_n/dt.

More generally, policy-conditioned assembly can return A(a), b(a) for
A(a) * v = b(a), with boundary contributions included. The iteration solves

    [A(a_n) + I/dt] * v_next = b(a_n) + v_n/dt.

Finite positive dt is artificial time for the stationary solve. With dt = Inf,
the pseudo-time terms vanish and the update is Howard policy iteration.
Implement the infinite-step case explicitly. Start with fixed dt; adaptive
selection can follow after benchmark validation.

Stop using the nonlinear residual evaluated with a policy greedy for the updated
value function. Record value changes and policy changes as additional diagnostics.
A small value update alone is insufficient when dt is small.

## Forward problem

Support standalone Kolmogorov forward problems through the same three stages,
and derive a discretized forward problem from a discretized HJB plus fixed policy.
Reuse its policy-conditioned generator and consistent boundary treatment.

For a conservative generator L acting on backward values, probability masses obey

    dmass/dt = transpose(L) * mass.

For a density represented with integration weights W, use
inv(W) * transpose(L) * W. Use mass internally and provide density conversion.
Check generator conservation, nonnegative off-diagonal entries, mass preservation,
and nonnegativity. Absorbing or killed processes require explicit treatment rather
than assuming conservation.

A stationary solve requires unit total mass. Nonunique invariant distributions
must produce a clear diagnostic. A transient solve takes initial mass and a
physical time interval; its time step differs from the backward pseudo-time step.

## Initial implementation milestones

- Stationary consumption-saving HJB: one continuous asset state and finite-state income.
- Structured-grid upwinding with state constraints and sparse policy evaluation.
- Semi-implicit iteration for finite dt and Howard iteration at dt = Inf.
- Stationary and transient forward solves under the computed policy.
- Verification using a known or manufactured solution, nonlinear residuals,
  grid refinement, agreement between iteration choices, and forward mass conservation.
- A documented end-to-end example before stabilizing the public API.

Equilibrium loops, finite-horizon HJBs, adaptive grids, and more general PDE
classes are subsequent extensions.
