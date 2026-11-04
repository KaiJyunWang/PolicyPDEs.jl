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

function _hjb_regime_index(problem, regime)
    index = findfirst(isequal(regime), problem.regimes)
    index === nothing && throw(ArgumentError("unknown regime label: $regime"))
    return index
end

function _check_hjb_spatial(value, dimension, name)
    if value isa Real && dimension == 1
        _check_hjb_scalar(value, name)
    else
        _check_hjb_vector(value, dimension, name)
    end
    return nothing
end

_hjb_coordinate(value::Real, k) = value
_hjb_coordinate(value, k) = value[k]

"""
    hjb_coefficients(problem, x, control, regime=first(problem.regimes))

Evaluate `discount`, `drift`, `diffusion`, and `flow` for one CTMC component
of an [`HJBProblem`](@ref), returning a named tuple with those four fields.
All callbacks receive `(x, control, regime)`. The control is passed unchanged.

`x` is a finite real scalar in one dimension or a nonempty tuple/vector of
finite reals. For a named-tuple domain, its axis count must match the state
dimension. Discount and flow must return finite real scalars. Drift and
diffusion must each return one finite real entry per continuous coordinate,
using tuples/vectors; a scalar is also accepted for one continuous dimension.

Diffusion entries are volatilities of independent coordinate Brownian motions,
so the second-order coefficients are `diffusion[k]^2 / 2`. Matrix-valued
diffusion and correlated state innovations are outside the initial scope.
Returns retain callback result types without copying or numeric conversion.
Transition rates are checked separately when [`hjb_residual`](@ref) evaluates
the switching term. This function evaluates a description, not a discretizer.
"""
function hjb_coefficients(problem::HJBProblem, x, control,
                          regime = first(problem.regimes))
    dimension = _hjb_dimension(problem, x)
    _hjb_regime_index(problem, regime)
    discount = problem.discount(x, control, regime)
    drift = problem.drift(x, control, regime)
    diffusion = problem.diffusion(x, control, regime)
    flow = problem.flow(x, control, regime)
    _check_hjb_scalar(discount, "discount")
    _check_hjb_spatial(drift, dimension, "drift")
    _check_hjb_spatial(diffusion, dimension, "diffusion")
    _check_hjb_scalar(flow, "flow")
    return (; discount, drift, diffusion, flow)
end

"""
    hjb_residual(problem, x, control, values, gradient, second_derivatives;
                 regime=first(problem.regimes))

Evaluate the fixed-control HJB residual for the selected CTMC component:

    flow - discount * values[i]
      + sum(drift[k] * gradient[k] + diffusion[k]^2 * second_derivatives[k] / 2)
      + sum(q_ij * (values[j] - values[i]) for j != i).

`values` is the full vector `v(x)` (or tuple) in `problem.regimes` order.
`gradient` and `second_derivatives` contain the selected component's first
and pure second derivatives, one per continuous coordinate. Scalars are also
accepted for those derivatives in one dimension. Mixed derivatives are not
used because the coordinate diffusion is diagonal. All entries must be finite
real scalars. Coefficients follow [`hjb_coefficients`](@ref).

Switching preserves `x`. Each off-diagonal rate is evaluated as
`problem.transition(x, control, source_label, destination_label)` using the
source component's control, and must be finite and nonnegative. Diagonal rates
are never requested. With `transition=nothing`, switching contributes zero.

The result is RHS minus LHS for a fixed control, not a maximization over
controls. Policy iteration will select a feasible control for each component;
the Bellman equation is `max_control(residual) = 0`. In particular, when
discounting depends on control, keep `-discount * values[i]` inside that maximum.
No derivatives are computed and no PDE is solved by this helper.
"""
function hjb_residual(problem::HJBProblem, x, control, values, gradient,
                      second_derivatives; regime = first(problem.regimes))
    coefficients = hjb_coefficients(problem, x, control, regime)
    dimension = _hjb_dimension(problem, x)
    index = _hjb_regime_index(problem, regime)
    _check_hjb_vector(values, length(problem.regimes), "values")
    _check_hjb_spatial(gradient, dimension, "gradient")
    _check_hjb_spatial(second_derivatives, dimension, "second_derivatives")
    value = values[index]
    residual = coefficients.flow - coefficients.discount * value
    for k in 1:dimension
        residual += _hjb_coordinate(coefficients.drift, k) * _hjb_coordinate(gradient, k)
        residual += _hjb_coordinate(coefficients.diffusion, k)^2 *
                    _hjb_coordinate(second_derivatives, k) / 2
    end
    if problem.transition !== nothing
        for (j, destination) in enumerate(problem.regimes)
            j == index && continue
            rate = problem.transition(x, control, regime, destination)
            _check_hjb_scalar(rate, "transition rate")
            rate >= 0 || throw(ArgumentError("off-diagonal transition rates must be nonnegative"))
            residual += rate * (values[j] - value)
        end
    end
    return residual
end
