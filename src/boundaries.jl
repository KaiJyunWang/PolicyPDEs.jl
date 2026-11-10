abstract type AbstractBoundary end
abstract type AbstractLinearBoundary <: AbstractBoundary end

"""
    StateConstraintBoundary(derivative)

Keep the HJB equation on the face and require drift to stay inside the domain.
`derivative` supplies a zero-normal-drift derivative, as a finite scalar,
vector/tuple, or callable `derivative(x)`. It is used only for the unavailable
outward finite difference: backward at a lower face, forward at an upper face.
The available inward difference is computed from values without alteration.
No derivative equation is imposed between the face and its neighboring node.

The supplied derivative must produce zero normal drift for every component.
Normal diffusion must vanish on this face; reflecting diffusions need a different
boundary condition. State constraints apply to every incident face at corners.
"""
struct StateConstraintBoundary{D} <: AbstractBoundary
    derivative::D
    function StateConstraintBoundary(derivative)
        if _is_boundary_constant(derivative)
            derivative isa Number ? _check_boundary_scalar(derivative, "derivative") :
            _check_boundary_vector(derivative, "derivative")
        end
        new{typeof(derivative)}(derivative)
    end
end

function _state_constraint_derivative(condition::StateConstraintBoundary, x, m)
    value = _is_boundary_constant(condition.derivative) ? condition.derivative :
        condition.derivative(x)
    return _boundary_vector(value, m, "state-constraint derivative")
end

function _check_boundary_scalar(value, name)
    value isa Real && isfinite(value) ||
        throw(ArgumentError("boundary coefficient $name must be finite and real"))
    return nothing
end

_is_boundary_vector(value) = value isa Union{Tuple, AbstractVector}
_is_boundary_constant(value) = value isa Number || _is_boundary_vector(value)

function _check_boundary_vector(value, name)
    _is_boundary_vector(value) ||
        throw(ArgumentError("$name must be a scalar, vector, or tuple"))
    value isa AbstractVector && Base.require_one_based_indexing(value)
    isempty(value) && throw(ArgumentError("boundary coefficient vectors must be nonempty"))
    for entry in value
        _check_boundary_scalar(entry, name)
    end
    return nothing
end

function _boundary_count(values, ncomponents)
    if ncomponents !== nothing
        ncomponents isa Integer && ncomponents > 0 ||
            throw(ArgumentError("ncomponents must be a positive integer"))
        return Int(ncomponents)
    end
    for value in values
        _is_boundary_vector(value) && return length(value)
    end
    return 1
end

function _boundary_vector(value, count, name)
    if value isa Real
        _check_boundary_scalar(value, name)
        return ntuple(_ -> value, count)
    end
    _check_boundary_vector(value, name)
    length(value) == count || throw(DimensionMismatch("$name must have $count entries"))
    return value
end

function _check_constant_boundary_coefficients(a, b, c)
    constants = filter(_is_boundary_constant, (a, b, c))
    for value in constants
        value isa Number ? _check_boundary_scalar(value, "constant") :
        _check_boundary_vector(value, "constant")
    end
    count = _boundary_count(constants, nothing)
    for value in constants
        _boundary_vector(value, count, "constant")
    end
    if _is_boundary_constant(a) && _is_boundary_constant(b)
        av, bv = _boundary_vector(a, count, "a"), _boundary_vector(b, count, "b")
        any(i -> iszero(av[i]) && iszero(bv[i]), 1:count) &&
            throw(ArgumentError("a and b cannot both be zero in any component"))
    end
    return nothing
end

"""
    LinearBoundary(a, b, c)
    LinearBoundary(; a, b, c)

Define the componentwise face equality `a[i] * partial_k(v_i) + b[i] * v_i = c[i]`.
Its coordinate k is determined by its position in [`RectangularBoundary`](@ref).

Coefficients are finite real scalars, vectors/tuples, or callable structs/functions
`coefficient(x)` returning scalars or vectors/tuples for all value components.
Scalars broadcast to the component count. Callbacks take the full continuous
state with no discrete input. Constants are checked without evaluating callbacks.
At evaluation, each component must have at least one of a and b nonzero.

The derivative follows increasing coordinate k on both lower and upper faces.
For outward-normal conditions, negate the derivative coefficient on lower faces.
Value (Dirichlet) conditions use a = 0, derivative (Neumann) conditions use b = 0,
and Robin conditions use both. The solver replaces the selected HJB row.
"""
struct LinearBoundary{A, B, C} <: AbstractLinearBoundary
    a::A
    b::B
    c::C
    function LinearBoundary(a, b, c)
        _check_constant_boundary_coefficients(a, b, c)
        return new{typeof(a), typeof(b), typeof(c)}(a, b, c)
    end
end
LinearBoundary(; a, b, c) = LinearBoundary(a, b, c)

"""
    NonlinearBoundary(a, b, c)
    NonlinearBoundary(; a, b, c)

Define the equality `a[i] * partial_k(v_i) + b[i] * v_i = c[i]`, with coefficients
depending on the current values and derivatives. Each coefficient is a finite
scalar, vector/tuple, or callback `coefficient(x, v, Dv, D2v)` returning a scalar
or component vector/tuple. Callbacks receive all value components; `Dv` and
`D2v` are coordinate-by-component first/pure-second derivatives, as in HJBProblem.
Inputs are read-only. Constants are checked without evaluating callbacks.

The solver evaluates these callbacks using unmodified centered/one-sided
finite differences of the current values, freezes the coefficients for a linear
row replacement, then refreshes them for the next step and stationary residual.
The derivative points toward increasing coordinate k on both faces. Convergence
is not guaranteed for arbitrary nonlinear boundary equations.
Use LinearBoundary for coefficients that depend only on x.
"""
struct NonlinearBoundary{A, B, C} <: AbstractBoundary
    a::A
    b::B
    c::C
    function NonlinearBoundary(a, b, c)
        _check_constant_boundary_coefficients(a, b, c)
        return new{typeof(a), typeof(b), typeof(c)}(a, b, c)
    end
end
NonlinearBoundary(; a, b, c) = NonlinearBoundary(a, b, c)

_is_equality_boundary(condition) = condition isa Union{LinearBoundary, NonlinearBoundary}

"""
    LinearInequalityBoundary(a, b, c)
    LinearInequalityBoundary(; a, b, c)

Define `a[i] * partial_k(v_i) + b[i] * v_i <= c[i]` componentwise on a face.
Coefficient and derivative conventions match [`LinearBoundary`](@ref).
Express >= by negating all coefficients; no sign normalization is performed.
This describes a constraint, not a condition imposed as an equality everywhere.
The Picard solver enforces this through orientation-aware active-set complementarity.
"""
struct LinearInequalityBoundary{A, B, C} <: AbstractLinearBoundary
    a::A
    b::B
    c::C
    function LinearInequalityBoundary(a, b, c)
        _check_constant_boundary_coefficients(a, b, c)
        return new{typeof(a), typeof(b), typeof(c)}(a, b, c)
    end
end
LinearInequalityBoundary(; a, b, c) = LinearInequalityBoundary(a, b, c)

function _face_tuple(faces)
    faces isa AbstractBoundary && return (faces,)
    faces isa Union{Tuple, AbstractVector} || throw(
        ArgumentError("faces must be a boundary condition or a tuple/vector of conditions"),
    )
    faces isa AbstractVector && Base.require_one_based_indexing(faces)
    return Tuple(faces)
end

"""
    RectangularBoundary(; lower, upper)
    RectangularBoundary(lower, upper)

Specify one linear/nonlinear equality, inequality, or state constraint per lower and upper
coordinate face. Entry `lower[k]` is on `x[k] = domain.lower[k]`; `upper[k]`
is on `x[k] = domain.upper[k]`. Both collections must be nonempty and have
the same length as the [`RectangularDomain`](@ref). The HJB constructor checks
the domain/boundary dimension match.

Collections are stored as tuples; a single condition is also accepted for a
one-dimensional interval. Different faces may use different condition types.
A face's coefficients can return different values for all PDE components.
At an edge or corner, each incident face has its own condition; consistency
and numerical enforcement are responsibilities of the discretizer/solver.

```julia
boundary = RectangularBoundary(
    lower = (LinearInequalityBoundary(-1, 0, -2), LinearBoundary(1, 0, 0)),
    upper = (LinearBoundary(1, 0, 0), LinearBoundary(0, 1, 0)),
)
```
"""
struct RectangularBoundary{L <: Tuple, U <: Tuple}
    lower::L
    upper::U
    function RectangularBoundary(lower, upper)
        lo, hi = _face_tuple(lower), _face_tuple(upper)
        isempty(lo) && throw(ArgumentError("at least one coordinate face pair is required"))
        length(lo) == length(hi) ||
            throw(DimensionMismatch("lower and upper face counts must match"))
        all(face -> face isa AbstractBoundary, (lo..., hi...)) || throw(
            ArgumentError(
                "each face must be a linear/nonlinear equality, inequality, or state-constraint boundary",
            ),
        )
        return new{typeof(lo), typeof(hi)}(lo, hi)
    end
end
RectangularBoundary(; lower, upper) = RectangularBoundary(lower, upper)
Base.length(boundary::RectangularBoundary) = length(boundary.lower)

_boundary_coefficient(value::Real, x) = value
_boundary_coefficient(value::Union{Tuple, AbstractVector}, x) = value
_boundary_coefficient(value, x) = value(x)

_boundary_coefficient(value::Real, x, v, Dv, D2v) = value
_boundary_coefficient(value::Union{Tuple, AbstractVector}, x, v, Dv, D2v) = value
_boundary_coefficient(value, x, v, Dv, D2v) = value(x, v, Dv, D2v)

"""
    boundary_coefficients(condition, x; ncomponents = nothing)

Evaluate a face's coefficients once at continuous state x, returning `(; a, b, c)`
as component vectors/tuples. Scalars broadcast. Specify ncomponents to match
the HJB value dimension; otherwise infer it from the first vector coefficient,
or use one if all coefficients are scalars. Check all lengths, finite real
entries, and nondegeneracy of a/b in each component. Callbacks receive only x.
"""
function boundary_coefficients(condition::AbstractLinearBoundary, x; ncomponents = nothing)
    raw = (
        _boundary_coefficient(condition.a, x),
        _boundary_coefficient(condition.b, x),
        _boundary_coefficient(condition.c, x),
    )
    return _checked_boundary_coefficients(raw, ncomponents)
end

function _checked_boundary_coefficients(raw, ncomponents)
    count = _boundary_count(raw, ncomponents)
    count > 0 ||
        throw(ArgumentError("boundary coefficients must have at least one component"))
    a, b, c = (_boundary_vector(raw[i], count, (:a, :b, :c)[i]) for i = 1:3)
    any(i -> iszero(a[i]) && iszero(b[i]), 1:count) &&
        throw(ArgumentError("a and b cannot both be zero in any component"))
    return (; a, b, c)
end

"""
    boundary_coefficients(condition::NonlinearBoundary, x, v, Dv, D2v)

Evaluate each coefficient once with the full state/value/derivative inputs.
Validate nonempty finite values, d-by-m derivative matrices, component lengths,
finite coefficient outputs, and nonzero a/b in each component. Scalar outputs
broadcast to the length of v. No geometry or differentiation is performed.
"""
function boundary_coefficients(
    condition::NonlinearBoundary,
    x,
    values,
    gradient,
    second_derivatives,
)
    dimension = x isa Real ? 1 : x isa Union{Tuple, AbstractVector} ? length(x) : 0
    dimension > 0 ||
        throw(ArgumentError("x must be a scalar or nonempty coordinate vector/tuple"))
    x isa Real ? _check_boundary_scalar(x, "x") : _check_boundary_vector(x, "x")
    _check_boundary_vector(values, "values")
    count = length(values)
    _check_hjb_matrix(gradient, dimension, count, "gradient")
    _check_hjb_matrix(second_derivatives, dimension, count, "second_derivatives")
    raw = (
        _boundary_coefficient(condition.a, x, values, gradient, second_derivatives),
        _boundary_coefficient(condition.b, x, values, gradient, second_derivatives),
        _boundary_coefficient(condition.c, x, values, gradient, second_derivatives),
    )
    return _checked_boundary_coefficients(raw, count)
end

"""
    boundary_residual(condition::NonlinearBoundary, v, Dv, D2v; x, axis = 1)

Evaluate the actual nonlinear equality residual using supplied values and full
derivative matrices. axis selects the face's increasing-coordinate derivative.
"""
function boundary_residual(
    condition::NonlinearBoundary,
    values,
    gradient,
    second_derivatives;
    x,
    axis = 1,
)
    a, b, c = boundary_coefficients(condition, x, values, gradient, second_derivatives)
    axis isa Integer && 1 <= axis <= size(gradient, 1) ||
        throw(ArgumentError("axis must be a valid coordinate index"))
    return [a[i] * gradient[axis, i] + b[i] * values[i] - c[i] for i in eachindex(values)]
end

"""
    boundary_residual(condition, values, derivatives; x)

Return the vector `a .* derivatives + b .* values - c` for one face.
Values and derivatives must be nonempty, equally sized vectors/tuples.
Derivatives refer to the face coordinate in the increasing-coordinate direction.
Residuals must be zero for equalities and nonpositive for inequalities.
No tolerance is chosen and no condition is numerically enforced by this helper.
"""
function boundary_residual(condition::AbstractLinearBoundary, values, derivatives; x)
    _check_boundary_vector(values, "values")
    _check_boundary_vector(derivatives, "derivatives")
    length(derivatives) == length(values) || throw(
        DimensionMismatch("values and derivatives must have the same component count"),
    )
    a, b, c = boundary_coefficients(condition, x; ncomponents = length(values))
    return [a[i] * derivatives[i] + b[i] * values[i] - c[i] for i in eachindex(values)]
end
