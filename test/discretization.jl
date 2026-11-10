function stencil_value(f, x, h; derivative = 1, direction = :central)
    stencil = fd_stencil(h; derivative, direction)
    return sum(
        w * f(x + offset * h) for (offset, w) in zip(stencil.offsets, stencil.weights)
    )
end

@testset "FiniteDifference grid specifications" begin
    @testset "Endpoint-inclusive Cartesian grids" begin
        domain = RectangularDomain([0.0, 0.5], [5.0, 1.5])
        grid = equidistant(domain, (6, 5))
        @test grid.domain === domain
        @test size(grid) == (6, 5)
        @test size(grid, 1) == 6
        @test size(grid, 3) == 1
        @test ndims(grid) == 2
        @test length(grid) == 30
        @test all(==(1.0), grid.steps[1])
        @test all(==(0.25), grid.steps[2])
        @test first(grid.coordinates[1]) == 0.0
        @test last(grid.coordinates[2]) == 1.5
        @test grid[1] == (0.0, 0.5)
        @test grid[2] == (1.0, 0.5)
        @test grid[7] == (0.0, 0.75)
        @test grid[end] == (5.0, 1.5)
        @test grid[CartesianIndex(2, 3)] == (1.0, 1.0)
        @test collect(grid)[7] == grid[7]
        @test all(x -> x in domain, grid)
        @test size(equidistant(domain, 4)) == (4, 4)
        counts = [3, 4]
        snapshot = equidistant(domain; npoints = counts)
        counts[1] = 99
        @test size(snapshot) == (3, 4)
        interval = equidistant(RectangularDomain(0, 5), 11)
        @test size(interval) == (11,)
        @test interval[2] == 0.5
        @test collect(interval) == collect(0.0:0.5:5.0)
        exact = equidistant(RectangularDomain(0 // 1, 1 // 2), 5)
        @test first(exact.coordinates[1]) == 0 // 1
        @test last(exact.coordinates[1]) == 1 // 2
        @test all(==(1 // 8), exact.steps[1])
        @test_throws BoundsError grid[0]
        @test_throws BoundsError grid[31]
        @test_throws BoundsError grid[CartesianIndex(7, 1)]
        @test_throws DimensionMismatch grid[CartesianIndex(1)]
        @test_throws DimensionMismatch equidistant(domain, (3,))
        for invalid in (1, 0, -1, true, 3.0, (3, 1), (3, 2.5))
            @test_throws ArgumentError equidistant(domain, invalid)
        end
        narrow = RectangularDomain(1.0, nextfloat(1.0))
        @test_throws ArgumentError equidistant(narrow, 3)
    end

    @testset "Stencil consistency and convergence" begin
        h = 1 // 10
        for direction in (:forward, :backward, :central)
            @test stencil_value(x -> 3 * x + 2, 1 // 2, h; direction) == 3
        end
        @test stencil_value(x -> x^2, 1 // 2, h) == 1
        @test stencil_value(x -> x^2, 1 // 2, h; derivative = 2) == 2
        x = 0.7
        for derivative in (1, 2)
            truth = derivative == 1 ? cos(x) : -sin(x)
            coarse = abs(stencil_value(sin, x, 0.2; derivative) - truth)
            fine = abs(stencil_value(sin, x, 0.1; derivative) - truth)
            @test 3.9 < coarse / fine < 4.1
        end
        # HJB drift directions give nonnegative off-diagonal generator weights.
        for drift in (-2.0, 2.0)
            first = fd_stencil(0.1; direction = drift > 0 ? :forward : :backward)
            second = fd_stencil(0.1; derivative = 2)
            weights = Dict(-1 => 0.0, 0 => 0.0, 1 => 0.0)
            for (o, w) in zip(first.offsets, first.weights)
                weights[o] += drift * w
            end
            for (o, w) in zip(second.offsets, second.weights)
                weights[o] += 0.3^2 * w / 2
            end
            @test weights[-1] >= 0 && weights[1] >= 0
            @test abs(sum(values(weights))) < 1e-12
        end
        for invalid in (0, -1, Inf, NaN, 1 + im)
            @test_throws ArgumentError fd_stencil(invalid)
        end
        @test_throws ArgumentError fd_stencil(0.1; derivative = 3)
        @test_throws ArgumentError fd_stencil(0.1; direction = :unknown)
        @test_throws ArgumentError fd_stencil(0.1; derivative = 2, direction = :forward)
        @test_throws ArgumentError fd_stencil(nextfloat(0.0))
    end

    @testset "Counts and explicit coordinate grids inside the method" begin
        for invalid in ((), (2, 5), (true, 3), (3, [0, 1, 2]), ([0, 1], [0, 1, 2]))
            @test_throws ArgumentError FiniteDifference(invalid)
        end
        @test_throws ArgumentError FiniteDifference(([0, 0, 1],))
        @test_throws ArgumentError FiniteDifference(([0, 1, Inf],))
        @test FiniteDifference(101).specification == (101,)
        @test FiniteDifference(6, 5).specification == (6, 5)
        @test FiniteDifference(([0, 1, 5],)).specification[1] == [0, 1, 5]
        bomb = (args...) -> error("discretize must not evaluate user callbacks")
        condition = LinearBoundary(bomb, 0, bomb)
        boundary = RectangularBoundary(
            lower = (condition, condition),
            upper = (condition, condition),
        )
        problem = HJBProblem(
            bomb,
            bomb,
            bomb,
            bomb;
            domain = RectangularDomain([0.0, 0.5], [5.0, 1.5]),
            boundary,
            transition = bomb,
        )
        method = FiniteDifference((6, 5))
        prepared = discretize(problem, method)
        @test prepared isa DiscretizedHJBProblem
        @test prepared.problem === problem
        @test prepared.grid.domain === problem.domain
        @test prepared.method === method
        @test prepared.method isa AbstractSpatialDiscretization
        @test size(prepared.grid) == (6, 5)
        @test prepared.stencils[1][2].forward.weights == (-1.0, 1.0)
        @test prepared.stencils[2][2].second.weights == (16.0, -32.0, 16.0)
        @test prepared.stencils[1][1].backward === nothing
        @test prepared.stencils[1][end].forward === nothing
        @test prepared.stencils[1][1].central === nothing
        @test prepared.stencils[1][end].second === nothing
        explicit = FiniteDifference((range(0.0, 5.0; length = 6), range(0.5, 1.5; length = 5)))
        @test discretize(problem, explicit).stencils == prepared.stencils
        @test_throws DimensionMismatch discretize(problem, FiniteDifference((6,)))
        @test_throws ArgumentError discretize(
            problem,
            FiniteDifference(([0, 1, 6], [0.5, 1, 1.5])),
        )
        @test_throws DimensionMismatch DiscretizedHJBProblem(
            problem,
            prepared.grid,
            FiniteDifference((5, 5)),
        )
        economic = discretize(two_income_problem(), FiniteDifference((101,)))
        @test size(economic.grid) == (101,)
        @test economic.grid[end] == 5.0
    end

    @testset "Nonuniform stencils and snapshot of supplied grids" begin
        problem = coupled_diffusion_problem()
        xgrid = [0.0, 0.1, 1.0, 5.0]
        ygrid = [0.5, 0.75, 1.5]
        method = FiniteDifference(xgrid, ygrid)
        xgrid[2] = 0.2
        prepared = discretize(problem, method)
        @test prepared.grid.coordinates[1][2] == 0.1
        @test size(prepared.grid) == (4, 3)
        @test prepared.grid.steps[1] ≈ [0.1, 0.9, 4.0]
        @test prepared.grid[5] == (0.0, 0.75)
        @test all(isapprox.(prepared.stencils[1][2].forward.weights, (-1 / 0.9, 1 / 0.9)))
        stencil = fd_stencil((1 // 10, 2 // 10); derivative = 2)
        point = 1 // 2
        @test sum(
            w * (point + (o == -1 ? -1 // 10 : o == 1 ? 2 // 10 : 0))^2 for
            (o, w) in zip(stencil.offsets, stencil.weights)
        ) == 2
        central = fd_stencil((1 // 10, 2 // 10))
        @test sum(
            w * (point + (o == -1 ? -1 // 10 : o == 1 ? 2 // 10 : 0))^2 for
            (o, w) in zip(central.offsets, central.weights)
        ) == 1
        @test fd_stencil((1 // 10, 1 // 10)) == fd_stencil(1 // 10)
        @test fd_stencil(method, 1 // 10) == fd_stencil(1 // 10)
        @test_throws ArgumentError fd_stencil((0.1,))
        @test_throws ArgumentError fd_stencil((0.1, 0.0))
        for axis = 1:2
            nodes = prepared.grid.coordinates[axis]
            for j = 2:(length(nodes) - 1)
                weights = prepared.stencils[axis][j]
                @test sum(
                    w * nodes[j + o]^2 for
                    (o, w) in zip(weights.central.offsets, weights.central.weights)
                ) ≈ 2 * nodes[j]
                @test sum(
                    w * nodes[j + o]^2 for
                    (o, w) in zip(weights.second.offsets, weights.second.weights)
                ) ≈ 2
            end
        end
        # Forward/backward offsets stay local even on unequal left/right steps.
        for drift in (-2.0, 2.0)
            first = fd_stencil((0.1, 0.3); direction = drift > 0 ? :forward : :backward)
            second = fd_stencil((0.1, 0.3); derivative = 2)
            weights = Dict(-1 => 0.0, 0 => 0.0, 1 => 0.0)
            for (o, w) in zip(first.offsets, first.weights)
                weights[o] += drift * w
            end
            for (o, w) in zip(second.offsets, second.weights)
                weights[o] += 0.3^2 * w / 2
            end
            @test weights[-1] >= 0 && weights[1] >= 0
            @test abs(sum(values(weights))) < 1e-12
        end
    end
end
