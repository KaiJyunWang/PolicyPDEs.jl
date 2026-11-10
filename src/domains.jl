function _bound_tuple(bounds)
    bounds isa Real && return (bounds,)
    bounds isa Union{Tuple, AbstractVector} ||
        throw(ArgumentError("bounds must be real scalars, tuples, or vectors"))
    bounds isa AbstractVector && Base.require_one_based_indexing(bounds)
    return Tuple(bounds)
end

"""
    RectangularDomain(lower, upper)
    RectangularDomain(; lower, upper)

The closed, axis-aligned rectangle `prod([lower[k], upper[k]])` in continuous
state space. Supply one finite real lower and upper bound per coordinate,
using nonempty tuples or vectors of the same length. Scalars are also accepted
for a one-dimensional interval. Each lower bound must be strictly below its
upper bound. Discrete value components do not appear in these bounds.

Bounds are stored as tuples, preserving entry types and taking a snapshot of
input vectors. `length(domain)` is the continuous dimension. Coordinate order
matches rows of drift, diffusion, and derivative matrices, and the order of
faces in [`RectangularBoundary`](@ref). Membership `x in domain` includes faces
and corners. A scalar state is accepted only for a one-dimensional domain.

```julia
domain = RectangularDomain(lower = [0.0, 0.5], upper = [5.0, 1.5])
```
"""
struct RectangularDomain{L <: Tuple, U <: Tuple}
    lower::L
    upper::U

    function RectangularDomain(lower, upper)
        lo, hi = _bound_tuple(lower), _bound_tuple(upper)
        isempty(lo) && throw(ArgumentError("domain must have at least one coordinate"))
        length(lo) == length(hi) ||
            throw(DimensionMismatch("lower and upper bounds must have equal length"))
        for k in eachindex(lo)
            for value in (lo[k], hi[k])
                value isa Real && isfinite(value) ||
                    throw(ArgumentError("domain bounds must be finite real scalars"))
            end
            lo[k] < hi[k] || throw(
                ArgumentError("each lower bound must be strictly below its upper bound"),
            )
        end
        return new{typeof(lo), typeof(hi)}(lo, hi)
    end
end

RectangularDomain(; lower, upper) = RectangularDomain(lower, upper)
Base.length(domain::RectangularDomain) = length(domain.lower)

function Base.in(x, domain::RectangularDomain)
    if x isa Real
        return length(domain) == 1 && domain.lower[1] <= x <= domain.upper[1]
    end
    x isa Union{Tuple, AbstractVector} || return false
    length(x) == length(domain) || return false
    x isa AbstractVector && Base.require_one_based_indexing(x)
    return all(
        k -> x[k] isa Real && domain.lower[k] <= x[k] <= domain.upper[k],
        eachindex(domain.lower),
    )
end
