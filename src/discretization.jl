"""
    AbstractSpatialDiscretization

Extension point for methods supplied to `discretize(problem, method)`.
Grid specifications belong to the method; iteration parameters belong to solve.
"""
abstract type AbstractSpatialDiscretization end

function _snapshot_fd_axis(axis; minimum = 3)
    axis isa Union{Tuple, AbstractVector} || throw(
        ArgumentError(
            "each coordinate grid must be a one-dimensional range, vector, or tuple",
        ),
    )
    axis isa AbstractVector && Base.require_one_based_indexing(axis)
    length(axis) >= minimum ||
        throw(ArgumentError("each coordinate grid requires at least $minimum points"))
    all(x -> x isa Real && isfinite(x), axis) ||
        throw(ArgumentError("grid coordinates must be finite real scalars"))
    all(j -> axis[j] < axis[j + 1], 1:(length(axis) - 1)) ||
        throw(ArgumentError("grid coordinates must be strictly increasing"))
    return axis isa AbstractVector && !(axis isa AbstractRange) ? copy(axis) : axis
end

"""
    FiniteDifference((n1, n2, ...))
    FiniteDifference((xgrid, ygrid, ...))
    FiniteDifference(xgrid, ygrid, ...)
    FiniteDifference(xgrid)

Classical finite differences, configured with either a tuple of integer point
counts (one per continuous coordinate) or explicit one-dimensional coordinate
grids. A single grid or integer is accepted for one dimension. Counts include
endpoints and require at least three points. Counts and coordinate grids cannot
be mixed. Explicit grids may be nonuniform, must be finite and strictly increasing,
and must include the problem's exact domain endpoints.

Mutable coordinate vectors are snapshotted. Treat method/grid data as read-only.
Use `discretize(problem, FiniteDifference((101, 51)))` for equidistant grids
built from the rectangular domain, or supply coordinate vectors/ranges.

The prepared stencils use actual neighboring spacings. First derivatives have
one-sided two-point and centered three-point forms; second derivatives have
three-point forms. Uniform centered stencils are second-order accurate.
No mixed derivatives, operator assembly, boundary enforcement, or solve is
implemented yet. For the backward HJB generator, positive drift uses forward
differences and negative drift uses backward differences.
"""
struct FiniteDifference{S} <: AbstractSpatialDiscretization
    specification::S
    function FiniteDifference(specification::Tuple)
        isempty(specification) &&
            throw(ArgumentError("provide coordinate grids or point counts"))
        if all(n -> n isa Integer && !(n isa Bool), specification)
            all(n -> 3 <= n <= typemax(Int), specification) || throw(
                ArgumentError(
                    "each coordinate needs at least three points, including endpoints",
                ),
            )
            stored = map(Int, specification)
        else
            all(axis -> axis isa Union{Tuple, AbstractVector}, specification) || throw(
                ArgumentError(
                    "provide either integer counts or coordinate grids, without mixing them",
                ),
            )
            stored = map(_snapshot_fd_axis, specification)
        end
        return new{typeof(stored)}(stored)
    end
end

FiniteDifference(axis::AbstractVector) = FiniteDifference((axis,))
FiniteDifference(n::Integer) = FiniteDifference((n,))
FiniteDifference(first, second, rest...) = FiniteDifference((first, second, rest...))

function _grid_counts(domain, npoints)
    if npoints isa Integer && !(npoints isa Bool)
        counts = ntuple(_ -> npoints, length(domain))
    elseif npoints isa Union{Tuple, AbstractVector}
        npoints isa AbstractVector && Base.require_one_based_indexing(npoints)
        counts = Tuple(npoints)
    else
        throw(ArgumentError("npoints must be an integer or tuple/vector of integers"))
    end
    length(counts) == length(domain) ||
        throw(DimensionMismatch("one point count is required per continuous coordinate"))
    all(n -> n isa Integer && !(n isa Bool) && 2 <= n <= typemax(Int), counts) || throw(
        ArgumentError("each coordinate needs at least two points including endpoints"),
    )
    return map(Int, counts)
end

"""
    TensorGrid(domain, coordinates)

Cartesian product of one-dimensional grids matching a rectangular domain.
Coordinates may be nonuniform. Bounds, coordinate order, and endpoint inclusion
are validated; mutable axes are copied. `coordinates[k]` is the physical axis,
`steps[k][j]` is the interval between nodes j and j+1, and `size(grid)` gives
the spatial counts. Value components are not grid axes.

Linear indexing/iteration uses Julia column-major order, coordinate 1 fastest.
Nodes are scalar states in 1D and tuples otherwise. The full tensor of states
is not materialized. Treat coordinates and steps as read-only.
"""
struct TensorGrid{D, A, N, H}
    domain::D
    coordinates::A
    counts::N
    steps::H
    function TensorGrid(domain::RectangularDomain, coordinates::Tuple)
        length(coordinates) == length(domain) ||
            throw(DimensionMismatch("one coordinate grid is required per domain dimension"))
        axes = map(axis -> _snapshot_fd_axis(axis; minimum = 2), coordinates)
        for k in eachindex(axes)
            first(axes[k]) == domain.lower[k] && last(axes[k]) == domain.upper[k] ||
                throw(ArgumentError("coordinate endpoints must match the domain bounds"))
        end
        counts = map(length, axes)
        foldl(Base.checked_mul, counts; init = 1)
        steps = map(axis -> [axis[j + 1] - axis[j] for j = 1:(length(axis) - 1)], axes)
        all(h -> all(x -> isfinite(x) && x > 0, h), steps) ||
            throw(ArgumentError("neighboring grid steps must be finite and positive"))
        return new{typeof(domain), typeof(axes), typeof(counts), typeof(steps)}(
            domain,
            axes,
            counts,
            steps,
        )
    end
end

"""
    equidistant(domain, npoints)
    equidistant(domain; npoints)

Build an endpoint-inclusive TensorGrid on the rectangle. An integer repeats on
all coordinates; a tuple/vector supplies one count per coordinate. The helper
requires at least two points; FiniteDifference requires at least three.
For the primary API, `FiniteDifference((n1, n2, ...))` builds the equidistant
grid automatically when passed to discretize.
"""
function equidistant(domain::RectangularDomain, npoints)
    counts = _grid_counts(domain, npoints)
    coordinates = ntuple(
        k -> range(domain.lower[k], domain.upper[k]; length = counts[k]),
        length(domain),
    )
    return TensorGrid(domain, coordinates)
end
equidistant(domain::RectangularDomain; npoints) = equidistant(domain, npoints)

Base.size(grid::TensorGrid) = grid.counts
Base.size(grid::TensorGrid, k::Integer) = k > 0 ? (k <= ndims(grid) ? grid.counts[k] : 1) :
    throw(ArgumentError("dimension must be positive"))
Base.ndims(grid::TensorGrid) = length(grid.domain)
Base.length(grid::TensorGrid) = prod(grid.counts)
Base.firstindex(grid::TensorGrid) = 1
Base.lastindex(grid::TensorGrid) = length(grid)
Base.IteratorSize(::Type{<:TensorGrid}) = Base.HasLength()
Base.IteratorEltype(::Type{<:TensorGrid}) = Base.EltypeUnknown()
function Base.getindex(grid::TensorGrid, index::CartesianIndex)
    length(Tuple(index)) == ndims(grid) ||
        throw(DimensionMismatch("node index dimension must match the grid"))
    checkbounds(CartesianIndices(size(grid)), index)
    state = ntuple(k -> grid.coordinates[k][index[k]], ndims(grid))
    return ndims(grid) == 1 ? state[1] : state
end
Base.getindex(grid::TensorGrid, index::Integer) = grid[CartesianIndices(size(grid))[index]]
Base.iterate(grid::TensorGrid, n = 1) = n > length(grid) ? nothing : (grid[n], n + 1)

function _positive_fd_step(h)
    h isa Real && isfinite(h) && h > 0 ||
        throw(ArgumentError("spacing must be finite, real, and positive"))
    return nothing
end

"""
    fd_stencil(spacing; derivative = 1, direction = :central)
    fd_stencil((left_step, right_step); derivative = 1, direction = :central)
    fd_stencil(method::FiniteDifference, spacing; ...)

Return (offsets, weights), with offsets measured in coordinate-node indices.
A scalar supplies a uniform step. For nonuniform grids the tuple supplies the
distance to each adjacent node. Forward uses the right step, backward the left.
Centered three-point weights differentiate the local quadratic interpolant.
First derivatives support :forward, :backward, :central; second derivatives
support :central. Uniform centered stencils are second order. On arbitrary
nonuniform grids the second derivative need not have second-order convergence.
No ghost nodes or boundary conditions are substituted.
"""
function fd_stencil(spacing; derivative = 1, direction = :central)
    if spacing isa Tuple
        length(spacing) == 2 || throw(ArgumentError("supply left and right spacings"))
        left, right = spacing
    else
        left = right = spacing
    end
    _positive_fd_step(left)
    _positive_fd_step(right)
    derivative isa Integer && !(derivative isa Bool) && derivative in (1, 2) ||
        throw(ArgumentError("derivative must be 1 or 2"))
    if derivative == 1
        if direction === :forward
            inverse = inv(right)
            result = (; offsets = (0, 1), weights = (-inverse, inverse))
        elseif direction === :backward
            inverse = inv(left)
            result = (; offsets = (-1, 0), weights = (-inverse, inverse))
        elseif direction === :central
            result = (;
                offsets = (-1, 0, 1),
                weights = (
                    -right / (left * (left + right)),
                    (right - left) / (left * right),
                    left / (right * (left + right)),
                ),
            )
        else
            throw(ArgumentError("direction must be :forward, :backward, or :central"))
        end
    else
        direction === :central ||
            throw(ArgumentError("second derivative supports only :central"))
        result = (;
            offsets = (-1, 0, 1),
            weights = (
                2 / (left * (left + right)),
                -2 / (left * right),
                2 / (right * (left + right)),
            ),
        )
    end
    all(isfinite, result.weights) ||
        throw(ArgumentError("finite-difference weights overflow at this spacing"))
    return result
end
fd_stencil(::FiniteDifference, spacing; kwargs...) = fd_stencil(spacing; kwargs...)

function _axis_stencils(axis, steps)
    return [
        begin
            forward = j < length(axis) ? fd_stencil(steps[j]; direction = :forward) : nothing
            backward = j > 1 ? fd_stencil(steps[j - 1]; direction = :backward) : nothing
            interior = 1 < j < length(axis)
            central = interior ? fd_stencil((steps[j - 1], steps[j])) : nothing
            second = interior ? fd_stencil((steps[j - 1], steps[j]); derivative = 2) : nothing
            (; forward, backward, central, second)
        end for j in eachindex(axis)
    ]
end

"""
    DiscretizedHJBProblem

Spatial preparation holding problem, grid, method, and stencils. Access
`stencils[k][j]` for coordinate k, node j, then .forward, .backward, .central,
or .second. Unavailable face stencils are nothing: there is no backward stencil
at the lower face, no forward at the upper face, and no centered stencil at
either face. Local spacings support nonuniform grids. Equality/inequality
boundary enforcement and operator assembly are performed by assemble_operator/solve.
Nodal values use node-by-component matrices in grid iteration order.
No callbacks are evaluated or value-component count inferred during preparation.
"""
struct DiscretizedHJBProblem{P, G, M, S}
    problem::P
    grid::G
    method::M
    stencils::S
    function DiscretizedHJBProblem(
        problem::HJBProblem,
        grid::TensorGrid,
        method::FiniteDifference,
    )
        problem.domain.lower == grid.domain.lower &&
            problem.domain.upper == grid.domain.upper ||
            throw(ArgumentError("grid bounds must match the problem domain"))
        all(n -> n >= 3, size(grid)) || throw(
            ArgumentError("FiniteDifference requires at least three points per coordinate"),
        )
        if all(x -> x isa Integer, method.specification)
            method.specification == size(grid) ||
                throw(DimensionMismatch("method counts must match the grid"))
        else
            method.specification == grid.coordinates ||
                throw(ArgumentError("method coordinate grids must match the grid"))
        end
        stencils = map(_axis_stencils, grid.coordinates, grid.steps)
        return new{typeof(problem), typeof(grid), typeof(method), typeof(stencils)}(
            problem,
            grid,
            method,
            stencils,
        )
    end
end

"""
    discretize(problem, FiniteDifference((n1, n2, ...)))
    discretize(problem, FiniteDifference((xgrid, ygrid, ...)))

Resolve counts to equidistant coordinate grids or validate supplied grids against
the problem rectangle, then prepare per-node classical FD stencils. Method data
contains the spatial specification, so no separate grid argument is required.
No economic coefficient or boundary callback is evaluated. Coefficient-dependent stencil
selection, operator assembly, and numerical boundary enforcement occur during solve.
"""
function discretize(problem::HJBProblem, method::FiniteDifference)
    length(method.specification) == length(problem.domain) ||
        throw(DimensionMismatch("one count or grid is required per continuous coordinate"))
    grid = if all(x -> x isa Integer, method.specification)
        equidistant(problem.domain, method.specification)
    else
        TensorGrid(problem.domain, method.specification)
    end
    return DiscretizedHJBProblem(problem, grid, method)
end
