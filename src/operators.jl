_sparse_identity(n) = spdiagm(0 => ones(Float64, n))

function _axis_operator(discrete, k, kind)
    axis = discrete.grid.coordinates[k]
    steps = discrete.grid.steps[k]
    n = length(axis)
    rows, columns, entries = Int[], Int[], Float64[]
    for j = 1:n
        stencil = getproperty(discrete.stencils[k][j], kind)
        if stencil === nothing
            if kind === :forward
                stencil = discrete.stencils[k][j].backward
            elseif kind === :backward
                stencil = discrete.stencils[k][j].forward
            elseif kind === :second
                if j == 1
                    local_stencil = fd_stencil((steps[1], steps[2]); derivative = 2)
                    stencil = (; offsets = (0, 1, 2), weights = local_stencil.weights)
                else
                    local_stencil = fd_stencil((steps[end - 1], steps[end]); derivative = 2)
                    stencil = (; offsets = (-2, -1, 0), weights = local_stencil.weights)
                end
            elseif j == 1
                h, g = steps[1], steps[2]
                stencil = (;
                    offsets = (0, 1, 2),
                    weights = (
                        -(2h + g) / (h * (h + g)),
                        (h + g) / (h * g),
                        -h / (g * (h + g)),
                    ),
                )
            else
                h, g = steps[end - 1], steps[end]
                stencil = (;
                    offsets = (-2, -1, 0),
                    weights = (
                        g / (h * (h + g)),
                        -(h + g) / (h * g),
                        (h + 2g) / (g * (h + g)),
                    ),
                )
            end
        end
        for (offset, weight) in zip(stencil.offsets, stencil.weights)
            push!(rows, j)
            push!(columns, j + offset)
            push!(entries, Float64(weight))
        end
    end
    result = sparse(rows, columns, entries, n, n)
    all(isfinite, nonzeros(result)) ||
        throw(ArgumentError("operator weights must fit finite Float64"))
    dropzeros!(result)
    return result
end

"""
    spatial_operators(discrete)

Build sparse forward, backward, central, and pure-second derivative operators.
Each field is a tuple indexed by coordinate, with node-by-node matrices lifted
by Kronecker products in coordinate-1-fastest order. Endpoint first derivatives
use available inward stencils; central endpoint derivatives use one-sided
three-point formulas. Endpoint second derivatives use the local quadratic.
Boundary rows are subsequently replaced by the selected face equations.
These endpoint closures do not guarantee generator monotonicity on inactive
inequality faces. Assembly and the initial solver use Float64 sparse matrices.
"""
function spatial_operators(discrete::DiscretizedHJBProblem)
    counts = size(discrete.grid)
    lift = (k, kind) -> begin
            before = prod(counts[1:(k - 1)])
            after = prod(counts[(k + 1):end])
            kron(
                _sparse_identity(after),
                kron(_axis_operator(discrete, k, kind), _sparse_identity(before)),
            )
        end
    return (;
        forward = ntuple(k -> lift(k, :forward), length(counts)),
        backward = ntuple(k -> lift(k, :backward), length(counts)),
        central = ntuple(k -> lift(k, :central), length(counts)),
        second = ntuple(k -> lift(k, :second), length(counts)),
    )
end

function _pack_coefficients(discrete, evaluated)
    n = length(discrete.grid)
    d = ndims(discrete.grid)
    m = length(first(evaluated).discount)
    all(c -> length(c.discount) == m, evaluated) ||
        throw(DimensionMismatch("value-component count must be fixed across grid nodes"))
    discount, flow = zeros(n, m), zeros(n, m)
    drift, diffusion = zeros(d, n, m), zeros(d, n, m)
    transition = zeros(n, m, m)
    for node = 1:n
        c = evaluated[node]
        discount[node, :] .= c.discount
        flow[node, :] .= c.flow
        drift[:, node, :] .= c.drift
        diffusion[:, node, :] .= c.diffusion
        c.transition === nothing || (transition[node, :, :] .= c.transition)
    end
    for data in (discount, flow, drift, diffusion, transition)
        all(isfinite, data) || throw(ArgumentError("coefficients must fit finite Float64"))
    end
    return (; discount, flow, drift, diffusion, transition)
end

struct _BoundaryRow
    row::Int
    node::Int
    component::Int
    axis::Int
    side::Symbol
    equality::Bool
    pde_sign::Float64
    a::Float64
    b::Float64
    c::Float64
    indices::Vector{Int}
    weights::Vector{Float64}
end

function _boundary_rows(discrete, m, operators; values = nothing)
    grid = discrete.grid
    n = length(grid)
    nonlinear = any(
        condition isa NonlinearBoundary for condition in
        (discrete.problem.boundary.lower..., discrete.problem.boundary.upper...)
    )
    gradient, second = nothing, nothing
    if nonlinear
        values === nothing &&
            throw(ArgumentError("values are required for nonlinear boundaries"))
        _check_hjb_matrix(values, n, m, "values")
        # Boundary callbacks must see the actual finite differences, not a
        # derivative already projected onto a frozen boundary equation.
        gradient, second = _grid_derivatives(discrete, values, operators, []; direction = :central)
    end
    rows = _BoundaryRow[]
    for (node, index) in enumerate(CartesianIndices(size(grid)))
        candidates = []
        for axis = 1:ndims(grid)
            side = index[axis] == 1 ? :lower :
                index[axis] == size(grid, axis) ? :upper : nothing
            side === nothing && continue
            condition = getproperty(discrete.problem.boundary, side)[axis]
            condition isa StateConstraintBoundary && continue
            coefficients = if condition isa NonlinearBoundary
                boundary_coefficients(
                    condition,
                    grid[node],
                    view(values, node, :),
                    view(gradient, :, node, :),
                    view(second, :, node, :),
                )
            else
                boundary_coefficients(condition, grid[node]; ncomponents = m)
            end
            push!(candidates, (; axis, side, condition, coefficients))
        end
        isempty(candidates) && continue
        for i = 1:m
            dirichlet = [
                face for face in candidates if
                _is_equality_boundary(face.condition) && iszero(face.coefficients.a[i])
            ]
            if length(dirichlet) > 1
                data = [face.coefficients.c[i] / face.coefficients.b[i] for face in dirichlet]
                all(v -> isapprox(v, first(data); atol = 1e-12, rtol = 1e-10), data) ||
                    throw(
                        ArgumentError(
                            "incompatible Dirichlet data at corner node $node, component $i",
                        ),
                    )
            end
            # Equality Dirichlet, then value inequality, then coordinate order.
            priority = face ->
                    iszero(face.coefficients.a[i]) ?
                    (_is_equality_boundary(face.condition) ? 0 : 1, face.axis) :
                    (2, face.axis)
            face = first(sort(candidates; by = priority))
            a, b, c = Float64(face.coefficients.a[i]),
            Float64(face.coefficients.b[i]),
            Float64(face.coefficients.c[i])
            entries = Dict(node => b)
            equality = _is_equality_boundary(face.condition)
            if !iszero(a)
                derivative = equality ? operators.central[face.axis] :
                    face.side === :lower ? operators.forward[face.axis] :
                    operators.backward[face.axis]
                columns, weights = findnz(derivative[node, :])
                for (column, weight) in zip(columns, weights)
                    entries[column] = get(entries, column, 0.0) + a * weight
                end
            end
            columns = sort(collect(keys(entries)))
            weights = [entries[column] for column in columns]
            all(isfinite, (a, b, c)) && all(isfinite, weights) ||
                throw(ArgumentError("boundary data must fit finite Float64"))
            own = entries[node]
            !equality &&
                iszero(own) &&
                throw(
                    ArgumentError(
                        "inequality boundary row must have a nonzero own-node coefficient",
                    ),
                )
            pde_sign = equality || own < 0 ? 1.0 : -1.0
            push!(
                rows,
                _BoundaryRow(
                    node + (i - 1) * n,
                    node,
                    i,
                    face.axis,
                    face.side,
                    equality,
                    pde_sign,
                    a,
                    b,
                    c,
                    columns .+ (i - 1) * n,
                    weights,
                ),
            )
        end
    end
    return rows
end

_boundary_value(row, values) = sum(w * values[j] for (j, w) in zip(row.indices, row.weights)) - row.c

function _active_boundaries(
    system,
    values;
    reference = values,
    shift = 0.0,
    previous = nothing,
    tolerance = 0.0,
)
    flat = vec(values)
    hjb = system.raw_b - system.raw_A * flat + shift * (vec(reference) - flat)
    return BitVector([
        row.equality ? true :
        begin
            boundary = _boundary_value(row, flat)
            oriented = row.pde_sign * hjb[row.row]
            abs(boundary - oriented) <= tolerance && previous !== nothing ? previous[j] :
            boundary > oriented
        end for (j, row) in enumerate(system.boundary_rows)
    ])
end

function _replace_boundary_rows(system, active; reference = nothing, shift = 0.0)
    count = length(system.raw_b)
    keep = ones(count)
    rhs = copy(system.raw_b)
    rows, columns, entries = Int[], Int[], Float64[]
    for (j, row) in enumerate(system.boundary_rows)
        active[j] || continue
        keep[row.row] = 0
        rhs[row.row] = row.c
        append!(rows, fill(row.row, length(row.indices)))
        append!(columns, row.indices)
        append!(entries, row.weights)
    end
    matrix = spdiagm(0 => keep) * system.raw_A + sparse(rows, columns, entries, count, count)
    if shift > 0
        matrix += spdiagm(0 => shift * keep)
        rhs .+= shift .* keep .* vec(reference)
    end
    dropzeros!(matrix)
    return matrix, rhs
end

function _system_residual(system, values; reference = values, shift = 0.0)
    flat = vec(values)
    residual = system.raw_b - system.raw_A * flat + shift * (vec(reference) - flat)
    boundary_norm = 0.0
    for row in system.boundary_rows
        boundary = _boundary_value(row, flat)
        residual[row.row] = row.equality ? boundary : max(row.pde_sign * residual[row.row], boundary)
        boundary_norm = max(boundary_norm, row.equality ? abs(boundary) : max(boundary, 0.0))
    end
    return (; residual, norm = norm(residual, Inf), boundary_norm)
end

"""
    assemble_operator(discrete, values; operators = spatial_operators(discrete))

Evaluate coefficients from node-by-component values and finite differences,
select upwind directions, and assemble sparse A*v = b with boundary rows replaced.
The result exposes coefficients, generator, raw_A/raw_b, A/b, and boundary data.
Values are stacked by component under vec, coordinate 1 fastest within a block.
No linear or nonlinear solve is performed.
"""
function assemble_operator(
    discrete::DiscretizedHJBProblem,
    values::AbstractMatrix;
    operators = spatial_operators(discrete),
)
    _check_hjb_matrix(values, length(discrete.grid), size(values, 2), "values")
    size(values, 2) > 0 || throw(ArgumentError("values must have at least one component"))
    rows = _boundary_rows(discrete, size(values, 2), operators; values)
    coefficients = _updated_coefficients(discrete, values, operators, rows)
    return _assemble_operator(discrete, coefficients; values, operators)
end

function _assemble_operator(
    discrete,
    coefficients;
    values,
    operators = spatial_operators(discrete),
)
    n, m = size(coefficients.discount)
    generator = spzeros(Float64, n * m, n * m)
    for i = 1:m
        spatial = spzeros(Float64, n, n)
        for k = 1:ndims(discrete.grid)
            drift = vec(coefficients.drift[k, :, i])
            spatial += spdiagm(0 => max.(drift, 0)) * operators.forward[k]
            spatial += spdiagm(0 => min.(drift, 0)) * operators.backward[k]
            spatial += spdiagm(0 => vec(coefficients.diffusion[k, :, i]) .^ 2 ./ 2) *
                operators.second[k]
        end
        block = sparse([i], [i], [1.0], m, m)
        generator += kron(block, spatial)
        for j = 1:m
            i == j && continue
            rates = spdiagm(0 => vec(coefficients.transition[:, i, j]))
            switching = sparse([i, i], [j, i], [1.0, -1.0], m, m)
            generator += kron(switching, rates)
        end
    end
    dropzeros!(generator)
    all(isfinite, nonzeros(generator)) ||
        throw(ArgumentError("generator entries must fit finite Float64"))
    raw_A = spdiagm(0 => vec(coefficients.discount)) - generator
    raw_b = vec(copy(coefficients.flow))
    boundary_rows = _boundary_rows(discrete, m, operators; values)
    if values === nothing
        any(row -> !row.equality, boundary_rows) &&
            throw(ArgumentError("values are required to select inequality boundary rows"))
        values = zeros(n, m)
    end
    _check_hjb_matrix(values, n, m, "values")
    system = (; raw_A, raw_b, generator, coefficients, boundary_rows, operators)
    active_boundary = _active_boundaries(system, values)
    A, b = _replace_boundary_rows(system, active_boundary)
    return merge(system, (; A, b, active_boundary))
end

function _grid_derivatives(
    discrete,
    values,
    operators,
    boundary_rows;
    drift = nothing,
    active = nothing,
    direction = nothing,
)
    n, m = size(values)
    d = ndims(discrete.grid)
    gradient, second = zeros(d, n, m), zeros(d, n, m)
    for i = 1:m, k = 1:d
        forward = operators.forward[k] * values[:, i]
        backward = operators.backward[k] * values[:, i]
        central = operators.central[k] * values[:, i]
        second[k, :, i] .= operators.second[k] * values[:, i]
        for node = 1:n
            gradient[k, node, i] = direction === :forward ? forward[node] :
                direction === :backward ? backward[node] :
                direction === :central || drift === nothing || iszero(drift[k, node, i]) ?
                central[node] : drift[k, node, i] > 0 ? forward[node] : backward[node]
        end
    end
    for (j, row) in enumerate(boundary_rows)
        iszero(row.a) && continue
        derivative = gradient[row.axis, row.node, row.component]
        violation = row.a * derivative + row.b * values[row.node, row.component] - row.c
        if row.equality || (active !== nothing && active[j]) || violation > 0
            gradient[row.axis, row.node, row.component] = ((row.c - row.b * values[row.node, row.component]) / row.a)
        end
    end
    # A state constraint supplies only the missing outward derivative. In
    # particular, never impose this derivative on the adjacent interior node.
    for (node, index) in enumerate(CartesianIndices(size(discrete.grid)))
        for axis = 1:d
            side = index[axis] == 1 ? :lower :
                index[axis] == size(discrete.grid, axis) ? :upper : nothing
            side === nothing && continue
            condition = getproperty(discrete.problem.boundary, side)[axis]
            condition isa StateConstraintBoundary || continue
            missing = (side === :lower && direction === :backward) ||
                (side === :upper && direction === :forward)
            missing || continue
            gradient[axis, node, :] .= _state_constraint_derivative(condition, discrete.grid[node], m)
        end
    end
    return gradient, second
end
