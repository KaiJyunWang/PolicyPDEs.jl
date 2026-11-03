# Design plan

This document records the intended design. Names and signatures remain provisional.

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
