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
supertype is required. Discount and flow return scalars; drift returns a scalar
or state vector. Diffusion returns volatility sigma, a scalar in one dimension
or a state-by-shock matrix in multiple dimensions. The second-order operator is
one half of sigma times its transpose contracted with the Hessian.

The immutable, parametrically typed container stores the four components plus
`domain`, `boundary`, `regimes`, and `transition` without copying, conversion,
or callback evaluation. Domain and boundary are required. Regimes must be a
nonempty tuple or vector of unique labels; their order determines PDE ordering.
Mutable inputs remain shared: callers must not mutate regime membership or
ordering after setup. The domain describes a common continuous state space;
boundary interpretation, component return shapes, and policy feasibility will
be defined and validated by the numerical stages.

Construction does not assemble an operator or solve the equations. Examples of
symbolic boundary descriptors demonstrate storage only. Initial guesses and
numerical settings belong to later stages.

## Markov-switching HJB systems

For regime i the equation is

    0 = max_a { f_i(x,a) - rho_i(x,a) * v_i(x)
                + b_i(x,a) dot grad(v_i)
                + (1/2) * tr(sigma_i * sigma_i' * Hessian(v_i))
                + sum(q_ij(x,a) * (v_j(x) - v_i(x)) for j != i) }.

`transition(x, a, i, j)` returns the finite, nonnegative rate **from i to j**,
for distinct labels i and j. It may depend on state and source-regime control.
Only off-diagonal rates are provided; the generator diagonal is minus their
outgoing sum. No diagonal callback is needed, including for a single PDE.
Switching preserves x. `transition = nothing` means zero rates; multiple
regimes then describe uncoupled HJBs. Numeric rate validation belongs to the
evaluation stage rather than probing user callbacks during construction.

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

For a discounted stationary HJB, write the discretized equation as

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
