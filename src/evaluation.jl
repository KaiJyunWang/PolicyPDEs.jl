function _check_hjb_scalar(value, name)
    value isa Real || throw(ArgumentError("$name must be a real scalar"))
    isfinite(value) || throw(ArgumentError("$name must be finite"))
    return nothing
end

function _check_hjb_vector(value, dimension, name)
    value isa Union{Tuple,AbstractVector} ||
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
    size(value) == (rows, columns) ||
        throw(DimensionMismatch("$name must have shape ($rows, $columns); got $(size(value))"))
    for entry in value
        _check_hjb_scalar(entry, name)
    end
    return nothing
end

function _hjb_dimension(problem, x)
    dimension = x isa Real ? 1 : x isa Union{Tuple,AbstractVector} ? length(x) : 0
    dimension > 0 || throw(ArgumentError("x must be a real scalar or a nonempty tuple/vector"))
    if x isa Real
        _check_hjb_scalar(x, "x")
    else
        _check_hjb_vector(x, dimension, "x")
    end
    if problem.domain isa NamedTuple && length(problem.domain) != dimension
        throw(DimensionMismatch("x must match the number of continuous axes in domain"))
    end
    return dimension
end

"""
    hjb_coefficients(problem, x, control)

Evaluate all five HJB primitives once at `(x, control)`, returning
`(; discount, drift, diffusion, flow, transition)` with the original result
types, without copying or conversion. No discrete-state input is required.

For `d` continuous coordinates and `m=length(problem.regimes)`, discount and
flow are length-`m` vectors/tuples, drift and diffusion are `d × m` matrices,
and transition is an `m × m` off-diagonal rate matrix or `nothing`. Check all
entries for finite real values, rates for nonnegativity, and transition
diagonals for exact zero. No scalar broadcasting or squeezing of singleton
matrix dimensions is performed.

`x` is a real scalar or nonempty tuple/vector of real coordinates. A named-tuple
domain must have the same number of axes. Control is passed unchanged; callbacks
define its representation. See [`HJBProblem`](@ref) for ordering and conventions.
"""
function hjb_coefficients(problem::HJBProblem, x, control)
    dimension = _hjb_dimension(problem, x)
    count = length(problem.regimes)
    discount = problem.discount(x, control)
    drift = problem.drift(x, control)
    diffusion = problem.diffusion(x, control)
    flow = problem.flow(x, control)
    transition = problem.transition === nothing ? nothing : problem.transition(x, control)
    _check_hjb_vector(discount, count, "discount")
    _check_hjb_matrix(drift, dimension, count, "drift")
    _check_hjb_matrix(diffusion, dimension, count, "diffusion")
    _check_hjb_vector(flow, count, "flow")
    if problem.transition !== nothing
        _check_hjb_matrix(transition, count, count, "transition")
        for i in 1:count, j in 1:count
            if i == j
                iszero(transition[i,j]) ||
                    throw(ArgumentError("transition diagonals must be zero; supply off-diagonal rates, not a generator"))
            else
                transition[i,j] >= 0 ||
                    throw(ArgumentError("off-diagonal transition rates must be nonnegative"))
            end
        end
    end
    return (; discount, drift, diffusion, flow, transition)
end

function _hjb_component_residual(coefficients, values, gradient, second_derivatives, i)
    residual = coefficients.flow[i] - coefficients.discount[i] * values[i]
    for k in axes(coefficients.drift, 1)
        residual += coefficients.drift[k,i] * gradient[k,i]
        residual += coefficients.diffusion[k,i]^2 * second_derivatives[k,i] / 2
    end
    if coefficients.transition !== nothing
        for j in eachindex(values)
            j == i && continue
            residual += coefficients.transition[i,j] * (values[j] - values[i])
        end
    end
    return residual
end

"""
    hjb_residual(problem, x, control, values, gradient, second_derivatives)

Return the length-`m` vector of fixed-control HJB residuals, RHS minus LHS,
for all components of `v(x)`. Evaluate each primitive once at `(x, control)`.

`values` is a length-`m` vector/tuple in regime order. `gradient` and
`second_derivatives` are `d × m` matrices: entries `[k,i]` contain the first
and pure second coordinate derivatives of component `i`. Shapes remain
explicit for single-coordinate or single-regime problems. All entries must be
finite reals. No mixed derivatives are used.

Component `i` is

    flow[i] - discount[i] * values[i]
      + sum(drift[k,i] * gradient[k,i]
            + diffusion[k,i]^2 * second_derivatives[k,i] / 2 for k in 1:d)
      + sum(transition[i,j] * (values[j] - values[i]) for j != i).

Switching preserves the continuous state and uses rows as source states.
With `transition=nothing`, the switching term vanishes. See
[`hjb_coefficients`](@ref) for coefficient validation.

This is evaluation for supplied controls, values, and derivatives. It does not
differentiate, optimize, or solve a PDE. Row-wise policy improvement maximizes
each component's complete residual over its feasible source control, including
the control-dependent discount term.
"""
function hjb_residual(problem::HJBProblem, x, control, values, gradient,
                      second_derivatives)
    coefficients = hjb_coefficients(problem, x, control)
    dimension, count = size(coefficients.drift)
    _check_hjb_vector(values, count, "values")
    _check_hjb_matrix(gradient, dimension, count, "gradient")
    _check_hjb_matrix(second_derivatives, dimension, count, "second_derivatives")
    return [_hjb_component_residual(coefficients, values, gradient, second_derivatives, i)
            for i in 1:count]
end
