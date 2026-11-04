# Design plan

This document records the intended design. Names and signatures remain provisional.

## Implemented: HJBProblem

The setup API accepts four components, either positionally or as keywords:

```julia
HJBProblem(discount, drift, diffusion, flow; domain, boundary,
           regimes = (1,), transition = nothing)
HJBProblem(; discount, drift, diffusion, flow, domain, boundary,
           regimes = (1,), transition = nothing)
```

Each component is a user-defined callable struct or function with signature
`component(x, a, i)`: continuous state, control, and regime label. No package
supertype is required. Discount and flow return scalars. Drift and diffusion
return tuples/vectors with one entry per continuous coordinate; scalars are
also accepted for a single continuous coordinate. Diffusion denotes volatility,
with independent coordinate Brownian motions and second-order coefficients
sigma_k^2/2. Matrix-valued diffusion and mixed derivative terms are outside the
initial supported equation class.

The immutable, parametrically typed container stores the four components plus
`domain`, `boundary`, `regimes`, and `transition` without copying, conversion,
or callback evaluation. Domain and boundary are required. Regimes must be a
nonempty tuple or vector of unique labels; their order determines PDE ordering.
Mutable inputs remain shared: callers must not mutate regime membership or
ordering after setup. The domain describes a common continuous state space.
Linear boundary definitions are provided below. Evaluation helpers check
coefficient shapes and rates; numerical enforcement and policy feasibility
await the discretization and solver stages.

Construction does not assemble an operator or solve the equations. Initial
guesses and numerical settings belong to later stages.

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

## Markov-switching HJB systems

For regime i the equation is

    0 = max_a { f_i(x,a) - rho_i(x,a) * v_i(x)
                + b_i(x,a) dot grad(v_i)
                + (1/2) * sum(sigma_ki(x,a)^2 * partial_kk(v_i) for k in 1:d)
                + sum(q_ij(x,a) * (v_j(x) - v_i(x)) for j != i) }.

`transition(x, a, i, j)` returns the finite, nonnegative rate **from i to j**,
for distinct labels i and j. It may depend on state and source-regime control.
Only off-diagonal rates are provided; the generator diagonal is minus their
outgoing sum. No diagonal callback is needed, including for a single PDE.
Switching preserves x. `transition = nothing` means zero rates; multiple
regimes then describe uncoupled HJBs. `hjb_residual` validates off-diagonal rates
at evaluation points without probing user callbacks during construction.

The continuous state is x in R^d and the unknown is v: R^d -> R^m, with
m = length(regimes). CTMC labels select the components v_i(x); they do not
add a continuous domain axis. Ordered state vectors/tuples follow the domain's
axis order; value-vector entries follow the order of regime labels. The sum
over k is over continuous coordinates, and the sum over j is over CTMC states.

The initial stochastic dynamics are

    dX_k = mu_ki(X,c) dt + sigma_ki(X,c) dW_k,

where the coordinate Brownian motions W_k are independent. No general
state-by-shock volatility matrix, cross-coordinate Brownian covariance, or
spatial reset upon a CTMC switch is included initially. All coefficients,
including discounting and CTMC rates, may depend on continuous state, source
regime, and source control.

`hjb_coefficients(problem, x, control, regime)` evaluates the four local
primitives and validates finite real outputs and coordinate dimensions.
`hjb_residual(problem, x, control, values, gradient, second_derivatives; regime)`
evaluates RHS minus LHS for one component using the full vector values=v(x)
and that component's gradient and pure second derivatives. It checks each
outgoing rate for finiteness and nonnegativity, never calls a diagonal rate,
and evaluates every coefficient using the supplied source control. These are
pointwise evaluation helpers; they do not compute derivatives, optimize a
policy, construct a grid, or solve a PDE. The Bellman equation is the maximum
of this residual over feasible controls. Control-dependent discounting must
stay inside that maximum.

With policy fixed, stack the regime values into a block linear system whose
generator contains both spatial and switching terms. For state/control-dependent
discounting, A is diag(rho(policy)) minus that full generator. The semi-implicit
step remains `(A + I/dt) * v_next = flow(policy) + v_current/dt`.

For the two-income-state setting in
[Achdou et al. (2022), REStud 89(1), 45-86](https://benjaminmoll.com/wp-content/uploads/2019/07/HACT.pdf),
the terms are lambda_1 * (v_2 - v_1) and lambda_2 * (v_1 - v_2).
[The runnable setup example](../examples/two_income_setup.jl) supplies four
callable structs and an asymmetric transition-rate struct. Parameters are
illustrative rather than a replication calibration. Arbitrary finite regime
counts and state/control-dependent rates use the same constructor. This API
describes switching among HJBs; the forward distribution equation remains a
separate planned API, and equilibrium feedback is not implemented.

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
