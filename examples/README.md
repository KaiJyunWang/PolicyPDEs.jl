# Examples

Start with consumption-saving under two income states. The model has one
continuous state (assets), two value functions (low/high income), and a
consumption choice for each income state.

[solve_two_income.jl](solve_two_income.jl) is a self-contained, top-level
walkthrough: parameters, primitives, policy, boundaries, grid, initial guess,
solve, and extraction of values, consumption, and saving are all in one script.

From the package root, run:

```sh
julia --project=. examples/solve_two_income.jl
```

Run each example in a fresh Julia session to keep its top-level names separate.
All setup, discretization, solve, and result variables stay available for inspection.

In a Julia REPL started with `julia --project=.`, the same workflow is:

```julia
using PolicyPDEs
include("examples/solve_two_income.jl")

# The script has already set up and solved the model. Inspect its variables:
solution.converged  # check before interpreting the results
solution.residual_norm
assets = collect(discrete.grid.coordinates[1])
value_low = solution.values[:, 1]
value_high = solution.values[:, 2]
consumption_low = (
    (model.income[1] .+ model.interest .* assets .- solution.coefficients.drift[1, :, 1])
)
consumption_high = (
    (model.income[2] .+ model.interest .* assets .- solution.coefficients.drift[1, :, 2])
)
saving_low = model.income[1] .+ model.interest .* assets .- consumption_low
```

`dt` is a solver parameter, not economic time. `solve(discrete; initial_value, dt = Inf)` evaluates
each frozen coefficient system fully (Howard iteration). Each coefficient function takes `(x, v, Dv, D2v)` directly. The example's flow
and drift share an internal log-consumption calculation; there is no control
callback in HJBProblem. The solver handles spatial upwinding.
Pass `show_trace = true` to `solve` to print aligned iteration and residual columns.
The two-income script enables this; the solver default is silent. End the solve
assignment with `;` to suppress display of the returned solution in the REPL.

At the lower asset bound, consumption cannot exceed available resources.
Both endpoints use `StateConstraintBoundary`: the HJB is retained and saving
may point inward. The zero-saving derivative is supplied only for the missing
outward difference. No equality is imposed between endpoint and neighbor values,
so saving at the upper endpoint and at the second asset node can be negative.

The two-income script includes optional plotting commands. Plots is not a
package dependency; install it in your own environment to use that section.
