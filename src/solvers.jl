function _validate_dt(dt)
    dt isa Real && dt > 0 && (isfinite(dt) || dt == Inf) ||
        throw(ArgumentError("dt must be positive and finite, or Inf"))
    converted = Float64(dt)
    isfinite(dt) &&
        !isfinite(converted) &&
        throw(ArgumentError("finite dt must fit Float64"))
    isfinite(converted) &&
        !isfinite(inv(converted)) &&
        throw(ArgumentError("inverse dt must fit finite Float64"))
    return converted
end

"""
    HJBSolution

Picard result. values is node-by-value-component; coefficients contains the
final frozen nodal fields: discount/flow are N-by-m, drift/diffusion d-by-N-by-m,
and transition N-by-m-by-m. No control vector is stored. discretization retains
the grid and model. converged/retcode/iterations report termination.
residual_norm measures the stationary residual after a fresh coefficient update.
boundary_residual_norm measures feasibility on the selected faces. value_change,
coefficient_change, residual_history, active_boundary, and linear_system expose
diagnostics. MaxIters returns a nonconverged result. Solver data use Float64.
"""
struct HJBSolution{D, C, S}
    discretization::D
    values::Matrix{Float64}
    coefficients::C
    converged::Bool
    retcode::Symbol
    iterations::Int
    residual_norm::Float64
    boundary_residual_norm::Float64
    value_change::Float64
    coefficient_change::Float64
    residual_history::Vector{Float64}
    active_boundary::BitVector
    linear_system::S
end

function _initial_matrix(discrete, initial_value)
    n = length(discrete.grid)
    if initial_value isa AbstractMatrix
        Base.require_one_based_indexing(initial_value)
        size(initial_value, 1) == n ||
            throw(DimensionMismatch("initial_value needs one row per grid node"))
        size(initial_value, 2) > 0 ||
            throw(ArgumentError("initial_value must have at least one value component"))
        all(v -> v isa Real && isfinite(v), initial_value) ||
            throw(ArgumentError("initial values must be finite and real"))
        values = Matrix{Float64}(initial_value)
    else
        rows = [initial_value(x) for x in discrete.grid]
        first(rows) isa Union{Tuple, AbstractVector} ||
            throw(ArgumentError("initial_value(x) must return a value vector"))
        m = length(first(rows))
        m > 0 || throw(ArgumentError("initial values must have at least one component"))
        values = zeros(n, m)
        for node = 1:n
            _check_hjb_vector(rows[node], m, "initial values")
            values[node, :] .= rows[node]
        end
    end
    all(isfinite, values) || throw(ArgumentError("initial values must fit finite Float64"))
    return values
end

function _local_coefficient_data(problem, x, values, gradient, second, forward, backward)
    coefficients = hjb_coefficients(problem, x, values, gradient, second)
    length(coefficients.discount) == length(values) ||
        throw(DimensionMismatch("coefficient outputs must match values"))
    score = zeros(length(values))
    for i in eachindex(values)
        score[i] = coefficients.flow[i] - coefficients.discount[i] * values[i]
        for k in axes(gradient, 1)
            mu = coefficients.drift[k, i]
            derivative = mu > 0 ? forward[k, i] : mu < 0 ? backward[k, i] : gradient[k, i]
            score[i] += mu * derivative + coefficients.diffusion[k, i]^2 * second[k, i] / 2
        end
        if coefficients.transition !== nothing
            score[i] += sum(
                coefficients.transition[i, j] * (values[j] - values[i]) for
                j in eachindex(values)
            )
        end
    end
    return (; coefficients, score)
end

# Choose one coordinate derivative conditional on the remaining derivatives.
# Only the (+, -) case admits a Hamiltonian comparison; (-, +) requires zero drift.
function _upwind_coordinate(problem, x, values, gradient, second, forward, backward, k, i)
    evaluate = g -> _local_coefficient_data(problem, x, values, g, second, forward, backward)
    forward_gradient = copy(gradient)
    backward_gradient = copy(gradient)
    forward_gradient[k, i] = forward[k, i]
    backward_gradient[k, i] = backward[k, i]
    saved_error = nothing
    try_candidate = g -> try
        evaluate(g)
    catch error
        error isa DomainError || rethrow()
        saved_error = error
        nothing
    end
    f = try_candidate(forward_gradient)
    b = try_candidate(backward_gradient)
    if f === nothing || b === nothing
        # An invalid directional coefficient evaluation cannot be used as a candidate.
        f !== nothing &&
            f.coefficients.drift[k, i] >= 0 &&
            return forward_gradient[k, i], false
        b !== nothing &&
            b.coefficients.drift[k, i] <= 0 &&
            return backward_gradient[k, i], false
        throw(saved_error)
    end
    mu_f = f.coefficients.drift[k, i]
    mu_b = b.coefficients.drift[k, i]
    if mu_f >= 0 && mu_b <= 0
        # Both directions are admissible. Compare each coefficient set's Hamiltonian.
        return f.score[i] >= b.score[i] ? (forward_gradient[k, i], false) :
               (backward_gradient[k, i], false)
    elseif mu_f >= 0 && mu_b > 0
        return forward_gradient[k, i], false
    elseif mu_f < 0 && mu_b <= 0
        return backward_gradient[k, i], false
    end

    # mu_f<0<mu_b: neither direction is admissible. Find coefficients whose
    # drift vanishes, rather than selecting the largest directional score.
    # Bisection assumes a continuous derivative-to-drift map.
    left, right = forward_gradient[k, i], backward_gradient[k, i]
    scale = max(1.0, abs(mu_f), abs(mu_b))
    for iteration = 1:100
        middle = left + (right - left) / 2
        trial = copy(gradient)
        trial[k, i] = middle
        candidate = evaluate(trial)
        mu = candidate.coefficients.drift[k, i]
        abs(mu) <= 1e-12 * scale && return middle, true
        if mu < 0
            left = middle
        else
            right = middle
        end
    end
    throw(
        ErrorException(
            "could not find a zero-drift coefficient evaluation for coordinate $k, component $i",
        ),
    )
end

function _local_upwind_coefficients(problem, x, values, center, second, forward, backward)
    d, m = size(center)
    gradient = Matrix(center)
    zero_drift = falses(d, m)
    # Repeat coordinate selections when a derivative affects multiple drifts.
    # The same signs must remain valid after the other derivatives are updated.
    for sweep = 1:100
        previous = copy(gradient)
        for i = 1:m, k = 1:d
            gradient[k, i], zero_drift[k, i] = _upwind_coordinate(
                problem,
                x,
                values,
                gradient,
                second,
                forward,
                backward,
                k,
                i,
            )
        end
        if d == 1 || isapprox(gradient, previous; atol = 1e-12, rtol = 1e-12)
            coefficients = hjb_coefficients(problem, x, values, gradient, second)
            actual = Matrix{Float64}(coefficients.drift)
            for index in findall(zero_drift)
                abs(actual[index]) <= 1e-10 || throw(
                    ErrorException("combined coefficients does not preserve zero drift"),
                )
            end
            actual[zero_drift] .= 0.0
            return merge(coefficients, (; drift = actual))
        end
    end
    throw(ErrorException("coordinate upwind coefficient selection did not converge"))
end

function _updated_coefficients(discrete, values, operators, boundary_rows; active = nothing)
    center, second = _grid_derivatives(
        discrete,
        values,
        operators,
        boundary_rows;
        direction = :central,
        active,
    )
    forward, _ = _grid_derivatives(
        discrete,
        values,
        operators,
        boundary_rows;
        direction = :forward,
        active,
    )
    backward, _ = _grid_derivatives(
        discrete,
        values,
        operators,
        boundary_rows;
        direction = :backward,
        active,
    )
    rows = [
        _local_upwind_coefficients(
            discrete.problem,
            discrete.grid[node],
            view(values, node, :),
            view(center, :, node, :),
            view(second, :, node, :),
            view(forward, :, node, :),
            view(backward, :, node, :),
        ) for node = 1:size(values, 1)
    ]
    for (node, index) in enumerate(CartesianIndices(size(discrete.grid)))
        for axis = 1:ndims(discrete.grid)
            side = index[axis] == 1 ? :lower :
                index[axis] == size(discrete.grid, axis) ? :upper : nothing
            side === nothing && continue
            condition = getproperty(discrete.problem.boundary, side)[axis]
            condition isa StateConstraintBoundary || continue
            missing = side === :lower ? backward : forward
            candidate = hjb_coefficients(
                discrete.problem,
                discrete.grid[node],
                view(values, node, :),
                view(missing, :, node, :),
                view(second, :, node, :),
            )
            for i in axes(values, 2)
                abs(candidate.drift[axis, i]) <= 1e-10 || throw(
                    ArgumentError(
                        "state-constraint derivative must give zero normal drift",
                    ),
                )
                iszero(candidate.diffusion[axis, i]) &&
                    iszero(rows[node].diffusion[axis, i]) || throw(
                    ArgumentError("state-constraint faces require zero normal diffusion"),
                )
                inward = side === :lower ? rows[node].drift[axis, i] : -rows[node].drift[axis, i]
                inward >= -1e-10 || throw(
                    ErrorException("selected drift points outside a state-constraint face"),
                )
            end
        end
    end
    return _pack_coefficients(discrete, rows)
end

function _coefficient_change(a, b)
    maximum(norm(vec(getproperty(a, name) - getproperty(b, name)), Inf) for name in keys(a))
end

function _coefficient_scale(a)
    max(1.0, maximum(norm(vec(getproperty(a, name)), Inf) for name in keys(a)))
end

function _linear_solution(A, b, algorithm, linear_atol, linear_rtol)
    solution = LinearSolve.solve(
        LinearSolve.LinearProblem(A, b),
        algorithm;
        abstol = linear_atol,
        reltol = linear_rtol,
    )
    values = solution.u
    all(isfinite, values) || throw(ErrorException("LinearSolve returned nonfinite values"))
    residual = norm(A * values - b, Inf)
    isfinite(residual) || throw(ErrorException("linear residual is nonfinite"))
    scale = opnorm(A, Inf) * norm(values, Inf) + norm(b, Inf)
    residual <= linear_atol + linear_rtol * scale ||
        throw(ErrorException("LinearSolve failed the checked linear residual: $residual"))
    return Vector{Float64}(values)
end

function _evaluate_policy(
    system,
    reference,
    dt,
    linear_algorithm;
    tolerance,
    max_active_iters,
    linear_atol,
    linear_rtol,
)
    shift = isinf(dt) ? 0.0 : inv(dt)
    current = copy(reference)
    previous = system.active_boundary
    for iteration = 1:max_active_iters
        active = _active_boundaries(
            system,
            current;
            reference,
            shift,
            previous,
            tolerance = tolerance / 10,
        )
        A, b = _replace_boundary_rows(system, active; reference, shift)
        flat = _linear_solution(A, b, linear_algorithm, linear_atol, linear_rtol)
        current = reshape(flat, size(reference))
        next_active = _active_boundaries(
            system,
            current;
            reference,
            shift,
            previous = active,
            tolerance = tolerance / 10,
        )
        residual = _system_residual(system, current; reference, shift)
        if next_active == active && residual.norm <= tolerance
            return Matrix(current), merge(system, (; A, b, active_boundary = active))
        end
        previous = next_active
    end
    throw(
        ErrorException(
            "boundary active set did not converge in $max_active_iters iterations",
        ),
    )
end

"""
    solve(
        discrete;
        initial_value,
        dt = Inf,
        maxiters = 500,
        atol = 1e-8,
        rtol = 1e-8,
        coefficient_tol = 1e-7,
        max_active_iters = 100,
        linear_algorithm = LinearSolve.UMFPACKFactorization(),
        linear_atol = 1e-12,
        linear_rtol = 1e-10,
        show_trace = false,
    )

Solve the stationary discretized HJB using Picard iteration. A finite positive
`dt` freezes coefficients and solves `(A + I / dt) * v_next = b + v_current / dt`
on HJB rows. Boundary rows are not pseudo-time shifted. The default `dt = Inf`
removes pseudo-time terms, giving full frozen-coefficient (Howard-type) iteration.

Supply `initial_value` as a node-by-component matrix or a callable
`initial_value(x)` returning all values. Node order matches
grid iteration. No automatic zero guess is used because coefficient callbacks may
require positive values or marginal values.

Each iteration differentiates current values, evaluates coefficient callbacks,
freezes their outputs, and solves the sparse linear system with LinearSolve.
Forward/backward coefficient evaluations determine drift signs. For each
coordinate/component: (+, -) compares Hamiltonians, (+, +) selects forward,
(-, -) selects backward, and (-, +) finds zero drift by bisection in the derivative.
Zero endpoints are admissible; Hamiltonian ties choose forward. Bisection needs
continuous coefficient functions and throws if no zero-drift evaluation is found.
Selections must stabilize across coordinates. A component's coefficients may
use all values (including switching gains), but its derivative dependence must
be local to that component's derivative column.
NonlinearBoundary coefficients are frozen from the current values and raw finite
differences, then refreshed before checking the stationary boundary residual.
Equality/binding derivative boundaries supply the normal derivative. Positive
drift uses forward differences, negative drift backward, and diffusion uses
pure centered second derivatives. Boundary rows replace HJB rows. Inequalities
use max(sign(q)*(HJB LHS-RHS), boundary residual) = 0, where q is the nonzero
own-node boundary coefficient.

Corners use equality Dirichlet first, then value inequalities, then coordinate
order for derivative ties; only the selected face is imposed at a corner.
Conflicting Dirichlet data throw an error. Endpoint PDE closures use inward
one-sided formulas; monotonicity is guaranteed by the upwind construction in
the interior, not for arbitrary inactive boundary inequalities.

Convergence requires a stationary residual <= atol+rtol*max(1, norm(values, Inf))
after a fresh coefficient update and a sufficiently small coefficient change. Small
pseudo-time value changes alone do not imply convergence. Coefficient iterations can fail or cycle; failures are not reported as success.
Linear residuals are checked after each solve. MaxIters returns diagnostics;
linear or boundary active-set failures throw errors. All solver data are Float64.

Set show_trace = true to print aligned iteration and stationary-residual columns
to stdout, one row per completed Picard iteration. The default is silent.
Tracing does not change the iteration or convergence criteria.
"""
function solve(
    discrete::DiscretizedHJBProblem;
    initial_value,
    dt = Inf,
    maxiters = 500,
    atol = 1e-8,
    rtol = 1e-8,
    coefficient_tol = 1e-7,
    max_active_iters = 100,
    linear_algorithm = LinearSolve.UMFPACKFactorization(),
    linear_atol = 1e-12,
    linear_rtol = 1e-10,
    show_trace = false,
)
    dt = _validate_dt(dt)
    show_trace isa Bool || throw(ArgumentError("show_trace must be a Bool"))
    maxiters isa Integer && !(maxiters isa Bool) && maxiters > 0 ||
        throw(ArgumentError("maxiters must be positive"))
    max_active_iters isa Integer && !(max_active_iters isa Bool) && max_active_iters > 0 ||
        throw(ArgumentError("max_active_iters must be positive"))
    for value in (atol, rtol, coefficient_tol, linear_atol, linear_rtol)
        value isa Real && isfinite(value) && value >= 0 ||
            throw(ArgumentError("tolerances must be finite and nonnegative"))
    end
    atol + rtol > 0 ||
        throw(ArgumentError("at least one nonlinear tolerance must be positive"))
    linear_atol + linear_rtol > 0 ||
        throw(ArgumentError("at least one linear tolerance must be positive"))
    values = _initial_matrix(discrete, initial_value)
    operators = spatial_operators(discrete)
    rows = _boundary_rows(discrete, size(values, 2), operators; values)
    active = falses(length(rows))
    history = Float64[]
    iteration_width = max(9, ndigits(maxiters))
    show_trace && println(lpad("Iteration", iteration_width), "  ", lpad("Residual", 24))
    for iteration = 1:maxiters
        coefficients = _updated_coefficients(discrete, values, operators, rows; active)
        system = _assemble_operator(discrete, coefficients; values, operators)
        tolerance = atol + rtol * max(1.0, norm(vec(values), Inf))
        next, evaluated = _evaluate_policy(
            system,
            values,
            dt,
            linear_algorithm;
            tolerance,
            max_active_iters,
            linear_atol,
            linear_rtol,
        )
        fresh_coefficients = _updated_coefficients(
            discrete,
            next,
            operators,
            evaluated.boundary_rows;
            active = evaluated.active_boundary,
        )
        checked = _assemble_operator(discrete, fresh_coefficients; values = next, operators)
        stationary = _system_residual(checked, next)
        push!(history, stationary.norm)
        value_change = norm(vec(next - values), Inf)
        coefficient_change = _coefficient_change(fresh_coefficients, coefficients)
        coefficient_scale = _coefficient_scale(fresh_coefficients)
        converged = stationary.norm <= atol + rtol * max(1.0, norm(vec(next), Inf)) &&
            coefficient_change <= coefficient_tol * coefficient_scale
        show_trace &&
            println(lpad(iteration, iteration_width), "  ", lpad(stationary.norm, 24))
        if converged || iteration == maxiters
            return HJBSolution(
                discrete,
                next,
                fresh_coefficients,
                converged,
                converged ? :Success : :MaxIters,
                iteration,
                stationary.norm,
                stationary.boundary_norm,
                value_change,
                coefficient_change,
                history,
                checked.active_boundary,
                checked,
            )
        end
        values = next
        rows = checked.boundary_rows
        active = checked.active_boundary
    end
    error("unreachable Picard termination")
end
