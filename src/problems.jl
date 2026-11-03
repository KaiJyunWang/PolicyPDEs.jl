"""
    HJBProblem(model; domain, boundary)

Describe a stationary Hamilton-Jacobi-Bellman problem before spatial
discretization. For example, a discounted HJB has the form

    rho * v(x) = max_a { reward(x, a) + L(a) * v(x) }.

# Arguments

- `model`: a user-defined object holding the economic primitives and parameters,
  such as discounting, rewards, dynamics, and admissible controls. No package
  supertype or fixed set of field names is required.
- `domain`: a description of the state space, independent of a numerical grid.
- `boundary`: an explicit description of the boundary conditions. There is no
  implicit default boundary condition.

The fields `model`, `domain`, and `boundary` retain the supplied objects and
their types, without copying or conversion. The problem is an immutable
container, but mutable objects stored in it remain mutable.

Construction only records the problem description: it does not evaluate the
model, validate mathematical well-posedness, or assemble numerical operators.
The accepted model, domain, and boundary interfaces will be defined by each
spatial discretization. Grids, initial value guesses, pseudo-time steps, and
solver tolerances do not belong to this constructor.

# Example

```julia
struct SavingsModel{T}
    discount::T
    interest::T
    income::T
end

# These descriptors illustrate problem storage; no discretizer is implemented yet.
model = SavingsModel(0.05, 0.03, 1.0)
domain = (; assets = (0.0, 100.0))
boundary = (; assets = (lower = :state_constraint, upper = :state_constraint))
problem = HJBProblem(model; domain, boundary)
```
"""
struct HJBProblem{M,D,B}
    model::M
    domain::D
    boundary::B
end

HJBProblem(model; domain, boundary) = HJBProblem(model, domain, boundary)
