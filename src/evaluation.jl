function _check_hjb_scalar(value, name)
    value isa Real || throw(ArgumentError("$name must be a real scalar"))
    isfinite(value) || throw(ArgumentError("$name must be finite"))
    return nothing
end

function _check_hjb_vector(value, dimension, name)
    value isa Union{Tuple, AbstractVector} ||
        throw(ArgumentError("$name must be a tuple or vector with $dimension entries"))
    value isa AbstractVector && Base.require_one_based_indexing(value)
    length(value) == dimension ||
        throw(DimensionMismatch("$name must have $dimension entries; got $(length(value))"))
    for entry in value
        _check_hjb_scalar(entry, name)
    end
    return nothing
end

function _check_hjb_matrix(value, rows, columns, name)
    value isa AbstractMatrix ||
        throw(ArgumentError("$name must be a $rows by $columns matrix"))
    Base.require_one_based_indexing(value)
    size(value) == (rows, columns) || throw(
        DimensionMismatch("$name must have shape ($rows, $columns); got $(size(value))"),
    )
    for entry in value
        _check_hjb_scalar(entry, name)
    end
    return nothing
end

function _hjb_dimension(problem, x)
    dimension = x isa Real ? 1 : x isa Union{Tuple, AbstractVector} ? length(x) : 0
    dimension > 0 ||
        throw(ArgumentError("x must be a real scalar or a nonempty tuple/vector"))
    if x isa Real
        _check_hjb_scalar(x, "x")
    else
        _check_hjb_vector(x, dimension, "x")
    end
    if length(problem.domain) != dimension
        throw(DimensionMismatch("x must match the rectangular domain dimension"))
    end
    x in problem.domain ||
        throw(DomainError(x, "state lies outside the rectangular domain"))
    return dimension
end

"""
    hjb_coefficients(problem, x, v, Dv, D2v)

Validate the state and value/derivative arrays, evaluate each supplied coefficient
once at (x, v, Dv, D2v), and validate its output. Return the original arrays in
(;discount, drift, diffusion, flow, transition), without copying. v determines m;
Dv/D2v must be d-by-m. Transition rates must be nonnegative off the zero diagonal.
"""
function hjb_coefficients(problem::HJBProblem, x, values, gradient, second_derivatives)
    dimension = _hjb_dimension(problem, x)
    values isa Union{Tuple, AbstractVector} ||
        throw(ArgumentError("values must be a vector or tuple"))
    count = length(values)
    count > 0 || throw(ArgumentError("values must have at least one component"))
    _check_hjb_vector(values, count, "values")
    _check_hjb_matrix(gradient, dimension, count, "gradient")
    _check_hjb_matrix(second_derivatives, dimension, count, "second_derivatives")
    discount = problem.discount(x, values, gradient, second_derivatives)
    drift = problem.drift(x, values, gradient, second_derivatives)
    diffusion = problem.diffusion(x, values, gradient, second_derivatives)
    flow = problem.flow(x, values, gradient, second_derivatives)
    transition = problem.transition === nothing ? nothing :
        problem.transition(x, values, gradient, second_derivatives)
    _check_hjb_vector(discount, count, "discount")
    _check_hjb_matrix(drift, dimension, count, "drift")
    _check_hjb_matrix(diffusion, dimension, count, "diffusion")
    _check_hjb_vector(flow, count, "flow")
    if problem.transition !== nothing
        _check_hjb_matrix(transition, count, count, "transition")
        for i = 1:count, j = 1:count
            if i == j
                iszero(transition[i, j]) || throw(
                    ArgumentError(
                        "transition diagonals must be zero; supply off-diagonal rates, not a generator",
                    ),
                )
            else
                transition[i, j] >= 0 || throw(
                    ArgumentError("off-diagonal transition rates must be nonnegative"),
                )
            end
        end
    end
    return (; discount, drift, diffusion, flow, transition)
end

function _hjb_component_residual(coefficients, values, gradient, second_derivatives, i)
    residual = coefficients.flow[i] - coefficients.discount[i] * values[i]
    for k in axes(coefficients.drift, 1)
        residual += coefficients.drift[k, i] * gradient[k, i]
        residual += coefficients.diffusion[k, i]^2 * second_derivatives[k, i] / 2
    end
    if coefficients.transition !== nothing
        for j in eachindex(values)
            j == i && continue
            residual += coefficients.transition[i, j] * (values[j] - values[i])
        end
    end
    return residual
end

"""
    hjb_residual(problem, x, v, Dv, D2v)

Evaluate each coefficient once, then return the componentwise HJB RHS minus LHS.
This pointwise helper does not discretize, select an upwind direction, or optimize.
"""
function hjb_residual(problem::HJBProblem, x, values, gradient, second_derivatives)
    coefficients = hjb_coefficients(problem, x, values, gradient, second_derivatives)
    return [
        _hjb_component_residual(coefficients, values, gradient, second_derivatives, i) for
        i in eachindex(values)
    ]
end

"""
    boundary_residual(problem, values, gradient; x, axis, side)
    boundary_residual(problem, values, gradient, second_derivatives; x, axis, side)

Evaluate all value-component residuals on one rectangular face. axis is a
coordinate index and side is :lower or :upper. x must lie in the closed
rectangle and on that face exactly. gradient is a d-by-m matrix; its selected
coordinate row supplies the increasing-coordinate derivatives. At corners,
call separately for each incident face. No PDE coefficient callbacks are
evaluated: the component count comes from values. For NonlinearBoundary supply
the full second-derivative matrix as the fourth positional argument.
"""
function boundary_residual(
    problem::HJBProblem,
    values,
    gradient;
    x,
    axis,
    side,
    second_derivatives = nothing,
)
    dimension = _hjb_dimension(problem, x)
    axis isa Integer && 1 <= axis <= dimension ||
        throw(ArgumentError("axis must be a coordinate index in 1:$dimension"))
    side in (:lower, :upper) || throw(ArgumentError("side must be :lower or :upper"))
    _check_boundary_vector(values, "values")
    _check_hjb_matrix(gradient, dimension, length(values), "gradient")
    point = x isa Real ? x : x[axis]
    endpoint = getproperty(problem.domain, side)[axis]
    point == endpoint || throw(ArgumentError("x must lie on the selected rectangular face"))
    condition = getproperty(problem.boundary, side)[axis]
    if second_derivatives !== nothing
        _check_hjb_matrix(
            second_derivatives,
            dimension,
            length(values),
            "second_derivatives",
        )
    end
    if condition isa NonlinearBoundary
        second_derivatives === nothing &&
            throw(ArgumentError("second derivatives are required for NonlinearBoundary"))
        return boundary_residual(condition, values, gradient, second_derivatives; x, axis)
    end
    return boundary_residual(condition, values, view(gradient, axis, :); x)
end

function boundary_residual(
    problem::HJBProblem,
    values,
    gradient,
    second_derivatives;
    x,
    axis,
    side,
)
    return boundary_residual(problem, values, gradient; x, axis, side, second_derivatives)
end
