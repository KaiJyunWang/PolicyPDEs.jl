abstract type AbstractLinearBoundary end

function _check_boundary_coefficient(value, name)
    value isa Real || throw(ArgumentError("boundary coefficient $name must be real"))
    isfinite(value) || throw(ArgumentError("boundary coefficient $name must be finite"))
    return nothing
end

function _check_boundary_coefficients(a, b, c)
    _check_boundary_coefficient(a, :a)
    _check_boundary_coefficient(b, :b)
    _check_boundary_coefficient(c, :c)
    iszero(a) && iszero(b) &&
        throw(ArgumentError("boundary coefficients a and b cannot both be zero"))
    return nothing
end

function _check_constant_boundary_coefficients(a, b, c)
    for (name, value) in ((:a, a), (:b, b), (:c, c))
        value isa Number && _check_boundary_coefficient(value, name)
    end
    if a isa Real && b isa Real && iszero(a) && iszero(b)
        throw(ArgumentError("boundary coefficients a and b cannot both be zero"))
    end
    return nothing
end

"""
    LinearEqualityBoundary(a, b, c)
    LinearEqualityBoundary(; a, b, c)

Define the linear boundary equality `a * v_x + b * v = c`.

Each coefficient is a finite real constant or a callable `coefficient(x, i)`
returning a finite real scalar at continuous state `x` and regime label `i`.
User-defined callable structs are supported. At least one of `a` and `b` must
be nonzero at each evaluation point. Inputs retain their types and are not
copied. Constant coefficients are checked at construction; callbacks are not
evaluated until `boundary_coefficients` or `boundary_residual` is called.

`v_x` is the derivative in the **increasing coordinate direction**, at both
the lower and upper endpoint, not the outward-normal derivative. The coordinate
is identified by the axis key in the enclosing boundary named tuple:

    boundary = (; assets = (
        lower = LinearEqualityBoundary(a=0, b=1, c=0),  # v = 0
        upper = LinearEqualityBoundary(a=1, b=0, c=0),  # v_x = 0
    ))

Coefficients may depend on the regime, so a single endpoint definition can
describe different conditions for coupled PDEs. Dirichlet conditions use `a=0`,
Neumann conditions use `b=0`, and Robin conditions use both coefficients.
These definitions do not assemble or enforce a numerical boundary condition.
See also [`LinearInequalityBoundary`](@ref), [`boundary_coefficients`](@ref),
and [`boundary_residual`](@ref).
"""
struct LinearEqualityBoundary{A,B,C} <: AbstractLinearBoundary
    a::A
    b::B
    c::C

    function LinearEqualityBoundary(a, b, c)
        _check_constant_boundary_coefficients(a, b, c)
        return new{typeof(a),typeof(b),typeof(c)}(a, b, c)
    end
end

LinearEqualityBoundary(; a, b, c) = LinearEqualityBoundary(a, b, c)

"""
    LinearInequalityBoundary(a, b, c)
    LinearInequalityBoundary(; a, b, c)

Define the linear boundary inequality `a * v_x + b * v <= c`.

Coefficients and the coordinate-derivative convention are the same as for
[`LinearEqualityBoundary`](@ref). The inequality always uses `<=`; express
`a * v_x + b * v >= c` by negating all three coefficients. Coefficients are
not normalized, so negative coefficients preserve the requested orientation.

For example, `v_x >= 2` is `LinearInequalityBoundary(a=-1, b=0, c=-2)`.
Its residual `-v_x + 2` is nonpositive when the inequality is satisfied.
An inequality describes a constraint, not an equality imposed at every point;
the numerical method must determine when it binds. No active-set,
complementarity, or boundary enforcement algorithm is implemented yet.
"""
struct LinearInequalityBoundary{A,B,C} <: AbstractLinearBoundary
    a::A
    b::B
    c::C

    function LinearInequalityBoundary(a, b, c)
        _check_constant_boundary_coefficients(a, b, c)
        return new{typeof(a),typeof(b),typeof(c)}(a, b, c)
    end
end

LinearInequalityBoundary(; a, b, c) = LinearInequalityBoundary(a, b, c)

_boundary_coefficient(value::Real, x, regime) = value
_boundary_coefficient(value, x, regime) = value(x, regime)

"""
    boundary_coefficients(boundary, x, regime=1)

Evaluate a linear equality or inequality boundary's coefficients, returning
`(; a, b, c)`. Constants are returned without conversion; callables receive
`(x, regime)`. Check that all results are finite real scalars and that `a` and
`b` are not both zero. Invalid coefficients throw `ArgumentError`.
"""
function boundary_coefficients(boundary::AbstractLinearBoundary, x, regime = 1)
    a = _boundary_coefficient(boundary.a, x, regime)
    b = _boundary_coefficient(boundary.b, x, regime)
    c = _boundary_coefficient(boundary.c, x, regime)
    _check_boundary_coefficients(a, b, c)
    return (; a, b, c)
end

"""
    boundary_residual(boundary, value, derivative; x, regime=1)

Return `a * derivative + b * value - c`, evaluating the coefficients at
`(x, regime)`. `derivative` is the increasing-coordinate derivative `v_x`
along the boundary's axis, including at the lower endpoint. The residual
must be zero for [`LinearEqualityBoundary`](@ref) and nonpositive for
[`LinearInequalityBoundary`](@ref). This function does not choose a numerical
tolerance or enforce the condition.
"""
function boundary_residual(boundary::AbstractLinearBoundary, value, derivative;
                           x, regime = 1)
    a, b, c = boundary_coefficients(boundary, x, regime)
    return a * derivative + b * value - c
end
