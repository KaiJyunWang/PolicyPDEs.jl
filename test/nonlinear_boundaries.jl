@testset "Nonlinear equality boundaries" begin
    @test !isdefined(PolicyPDEs, :LinearEqualityBoundary)
    @test LinearBoundary(0, 1, 2) isa LinearBoundary

    @testset "Full coefficient inputs and pointwise residuals" begin
        calls = zeros(Int, 3)
        seen = []
        a = (x, v, Dv, D2v) -> begin
            calls[1] += 1
            push!(seen, (x, v, Dv, D2v))
            [1 + v[1], 2 + v[2]]
        end
        b = (x, v, Dv, D2v) -> begin
            calls[2] += 1
            push!(seen, (x, v, Dv, D2v))
            Dv[1, :] .^ 2
        end
        c = (x, v, Dv, D2v) -> begin
            calls[3] += 1
            push!(seen, (x, v, Dv, D2v))
            D2v[2, :] .+ x[1]
        end
        condition = @inferred NonlinearBoundary(; a, b, c)
        @test calls == zeros(Int, 3)
        x = (0.5, 0.75)
        v = [1.0, 2.0]
        Dv = [2.0 3.0; 4.0 5.0]
        D2v = [-1.0 -2.0; -3.0 -4.0]
        coefficients = boundary_coefficients(condition, x, v, Dv, D2v)
        @test coefficients == (; a = [2.0, 4.0], b = [4.0, 9.0], c = [-2.5, -3.5])
        @test calls == ones(Int, 3)
        @test all(
            args -> args[1] === x && args[2] === v && args[3] === Dv && args[4] === D2v,
            seen,
        )
        fill!(calls, 0)
        @test boundary_residual(condition, v, Dv, D2v; x, axis = 2) == [14.5, 41.5]
        @test calls == ones(Int, 3)
        broadcast = NonlinearBoundary(1, (x, v, Dv, D2v) -> 2, [3, 4])
        @test boundary_coefficients(broadcast, x, v, Dv, D2v) == (; a = (1, 1), b = (2, 2), c = [3, 4])
        bomb = (args...) -> error("PDE callbacks are not needed for boundary evaluation")
        problem = HJBProblem(
            bomb,
            bomb,
            bomb,
            bomb;
            domain = RectangularDomain((0.0, 0.75), (1.0, 2.0)),
            boundary = RectangularBoundary(
                lower = (LinearBoundary(0, 1, 0), condition),
                upper = (LinearBoundary(0, 1, 0), LinearBoundary(0, 1, 0)),
            ),
        )
        @test boundary_residual(problem, v, Dv, D2v; x, axis = 2, side = :lower) == [14.5, 41.5]
        @test_throws ArgumentError boundary_residual(
            problem,
            v,
            Dv;
            x,
            axis = 2,
            side = :lower,
        )
        @test_throws ArgumentError boundary_residual(condition, v, Dv, D2v; x, axis = 3)
        @test_throws DimensionMismatch boundary_coefficients(
            condition,
            x,
            v,
            zeros(1, 2),
            D2v,
        )
        @test_throws DimensionMismatch boundary_coefficients(
            condition,
            x,
            v,
            Dv,
            zeros(2, 1),
        )
        @test_throws ArgumentError boundary_coefficients(condition, x, [NaN, 2.0], Dv, D2v)
        @test_throws ArgumentError NonlinearBoundary(0, 0, 1)
        @test_throws ArgumentError NonlinearBoundary(Inf, 1, 1)
        @test_throws ArgumentError boundary_coefficients(
            NonlinearBoundary((args...) -> 0, 0, 1),
            x,
            v,
            Dv,
            D2v,
        )
        @test_throws DimensionMismatch boundary_coefficients(
            NonlinearBoundary(1, 0, [1, 2, 3]),
            x,
            v,
            Dv,
            D2v,
        )
        @test_throws ArgumentError boundary_coefficients(
            NonlinearBoundary(1, 0, (args...) -> NaN),
            x,
            v,
            Dv,
            D2v,
        )
    end

    @testset "Frozen rows and actual nonlinear convergence" begin
        exact = x -> 1 + x
        lower = NonlinearBoundary(0, 1, (x, v, Dv, D2v) -> 0.5 * v[1] + 0.5 * exact(x))
        upper = NonlinearBoundary(
            (x, v, Dv, D2v) -> 1 + 0.05 * tanh(v[1])^2,
            (x, v, Dv, D2v) -> 1 + 0.01 * tanh(Dv[1, 1])^2 + 0.01 * tanh(D2v[1, 1])^2,
            (x, v, Dv, D2v) ->
                (1 + 0.05 * tanh(exact(x))^2) + (1 + 0.01 * tanh(1.0)^2) * exact(x),
        )
        problem = HJBProblem(;
            discount = (args...) -> [1.0],
            flow = (x, v, Dv, D2v) -> [exact(x) - 0.1],
            drift = (args...) -> fill(0.1, 1, 1),
            diffusion = (args...) -> fill(0.2, 1, 1),
            domain = RectangularDomain(0, 1),
            boundary = RectangularBoundary(; lower, upper),
        )
        discrete = discretize(problem, FiniteDifference((9,)))
        zero_system = assemble_operator(discrete, zeros(9, 1))
        one_system = assemble_operator(discrete, ones(9, 1))
        @test zero_system.b[1] == 0.5
        @test one_system.b[1] == 1.0
        stopped = solve(discrete; initial_value = zeros(9, 1), maxiters = 1)
        @test !stopped.converged
        @test stopped.boundary_residual_norm > 0.2
        for dt in (Inf, 2.0)
            solution = solve(
                discrete;
                dt,
                initial_value = zeros(9, 1),
                maxiters = 300,
                atol = 1e-10,
                rtol = 0,
            )
            @test solution.converged
            @test solution.iterations > 1
            @test isapprox(
                solution.values[:, 1],
                exact.(collect(discrete.grid));
                atol = 2e-8,
            )
            @test solution.boundary_residual_norm < 1e-9
            ops = spatial_operators(discrete)
            gradient, second = PolicyPDEs._grid_derivatives(
                discrete,
                solution.values,
                ops,
                [];
                direction = :central,
            )
            for (node, condition) in ((1, lower), (9, upper))
                actual = boundary_residual(
                    condition,
                    solution.values[node, :],
                    gradient[:, node, :],
                    second[:, node, :];
                    x = discrete.grid[node],
                )
                @test norm(actual, Inf) < 1e-9
            end
        end
    end

    @testset "Nonlinear Dirichlet corner priority and conflicts" begin
        dynamic = NonlinearBoundary(0, 1, (x, v, Dv, D2v) -> 1 + 0.5 * v[1])
        neumann = LinearBoundary(1, 0, 0)
        make_problem = boundary -> HJBProblem(;
                discount = (args...) -> [1.0],
                flow = (args...) -> [1.0],
                drift = (args...) -> zeros(2, 1),
                diffusion = (args...) -> zeros(2, 1),
                domain = RectangularDomain((0, 0), (1, 1)),
                boundary,
            )
        problem = make_problem(
            RectangularBoundary(lower = (dynamic, neumann), upper = (neumann, neumann)),
        )
        system = assemble_operator(discretize(problem, FiniteDifference((3, 3))), zeros(9, 1))
        @test system.A[1, 1] == 1.0
        @test count(!iszero, system.A[1, :]) == 1
        @test system.b[1] == 1.0
        conflict = make_problem(
            RectangularBoundary(
                lower = (dynamic, LinearBoundary(0, 1, 2)),
                upper = (neumann, neumann),
            ),
        )
        @test_throws ArgumentError assemble_operator(
            discretize(conflict, FiniteDifference((3, 3))),
            zeros(9, 1),
        )
    end
end
