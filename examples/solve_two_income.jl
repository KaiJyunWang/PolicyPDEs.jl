using PolicyPDEs

# Consumption-saving with log utility and low/high income:
# rho*v_j(a) = max_c {log(c_j) + (y_j+r*a-c_j)*v_j'(a)
#                    + lambda_jk*(v_k(a)-v_j(a))}, j != k.
# Illustrative parameters and upper truncation, not a replication.

# 1. Parameters. Component 1 is low income; component 2 is high income.
Base.@kwdef struct TwoIncomeModel
    income::Vector{Float64} = [0.1, 0.3]
    interest::Float64 = 0.02
    discount::Float64 = 0.03
    low_to_high::Float64 = 0.02
    high_to_low::Float64 = 0.03
    asset_min::Float64 = 0.0
    asset_max::Float64 = 5.0
end

const model = TwoIncomeModel()

# 2. Coefficients are functions of (assets, values, first derivatives,
# pure second derivatives). There is no control callback in the package API.
resources(a) = model.income .+ model.interest * a

# This model-specific helper is shared by flow and drift. Log utility gives
# c_j = 1/v_j'. It is an implementation detail of these coefficient functions.
function consumption_from_derivative(a, Dv)
    marginal_value = Dv[1, :]
    all(>(0), marginal_value) ||
        throw(DomainError(marginal_value, "log utility requires positive marginal values"))
    c = 1 ./ marginal_value
    return c
end

discount(a, v, Dv, D2v) = fill(model.discount, 2)
flow(a, v, Dv, D2v) = log.(consumption_from_derivative(a, Dv))
drift(a, v, Dv, D2v) = reshape(resources(a) .- consumption_from_derivative(a, Dv), 1, 2)
diffusion(a, v, Dv, D2v) = zeros(1, 2)
# Source-by-destination off-diagonal rates, with zero diagonal.
transition(a, v, Dv, D2v) = [0.0 model.low_to_high; model.high_to_low 0.0]

# 3. Keep the HJB at both endpoints and prevent outward saving.
# The zero-drift derivative 1/(y_j+r*a) supplies only the missing outward
# difference. The inward difference comes from v, so saving need not be zero
# at the upper endpoint, and no derivative equation distorts the second node.
domain = RectangularDomain(model.asset_min, model.asset_max)
boundary = RectangularBoundary(
    lower = StateConstraintBoundary(a -> 1 ./ resources(a)),
    upper = StateConstraintBoundary(a -> 1 ./ resources(a)),
)
problem = HJBProblem(; discount, flow, drift, diffusion, transition, domain, boundary)

# 4. Discretize assets, including both endpoints.
discrete = discretize(problem, FiniteDifference((201,)))
# For a custom grid instead: FiniteDifference((asset_grid,)).

# 5. Start from consuming current income plus interest forever.
# This increasing initial guess gives positive marginal values.
initial_value(a) = log.(model.income .+ model.interest * a) ./ model.discount

# 6. Evaluate coefficients, freeze them, and solve the linear HJB.
# dt is a numerical pseudo-time step. Use dt = Inf for Howard iteration.
solution = solve(discrete; dt = Inf, initial_value, maxiters = 200, show_trace = true);



# 7. Rows are asset nodes; columns are low/high income.
assets = collect(solution.discretization.grid.coordinates[1])
value = solution.values
consumption = [
    resources(a)[j] - solution.coefficients.drift[1, i, j] for
    (i, a) in enumerate(assets), j = 1:2
]
saving = [
    model.income[j] + model.interest * a - consumption[i, j] for
    (i, a) in enumerate(assets), j = 1:2
]

# Plot the results.
using Plots

labels = ["low income" "high income"]
plot(assets, value; label = labels, xlabel = "assets", ylabel = "value")
plot(assets, consumption; label = labels, xlabel = "assets", ylabel = "consumption")
plot(assets, saving; label = labels, xlabel = "assets", ylabel = "saving")
