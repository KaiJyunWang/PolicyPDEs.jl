function quadratic_switching_problem()
    exact = x -> [1 + x^2, 2 + 2x^2]
    rates = x -> [0.0 0.2+0.1 * x; 0.3+0.05 * x 0.0]
    return HJBProblem(;
        discount = (x, v, Dv, D2v) -> [1.0, 2.0],
        drift = (x, v, Dv, D2v) -> zeros(1, 2),
        diffusion = (x, v, Dv, D2v) -> [0.3 0.4],
        flow = (x, v, Dv, D2v) -> begin
            v = exact(x)
            q = rates(x)
            [v[1] - 0.09 - q[1, 2] * (v[2] - v[1]), 2v[2] - 0.32 - q[2, 1] * (v[1] - v[2])]
        end,
        transition = (x, v, Dv, D2v) -> rates(x),
        domain = RectangularDomain(0.0, 1.0),
        boundary = RectangularBoundary(
            lower = LinearBoundary(0, 1, exact),
            upper = LinearBoundary(0, 1, exact),
        ),
    )
end

@testset "Sparse policy assembly and Picard solve" begin
    @testset "Four drift-sign cases for direct coefficients" begin
        function directional_problem(mu, u)
            HJBProblem(;
                discount = (x, v, Dv, D2v) -> [1.0],
                flow = (x, v, Dv, D2v) -> [u(Dv[1, 1])],
                drift = (x, v, Dv, D2v) -> reshape([mu(Dv[1, 1])], 1, 1),
                diffusion = (x, v, Dv, D2v) -> zeros(1, 1),
                domain = RectangularDomain(0, 1),
                boundary = test_boundary(1),
            )
        end
        select = (problem, f, b) -> PolicyPDEs._local_upwind_coefficients(
                problem,
                0.5,
                [0.0],
                reshape([(f + b) / 2], 1, 1),
                zeros(1, 1),
                reshape([f], 1, 1),
                reshape([b], 1, 1),
            )
        @test select(directional_problem(c -> c - 2, c -> 100c), 3.0, 1.0).flow == [300.0]
        @test select(directional_problem(c -> c - 2, c -> -100c), 3.0, 1.0).flow == [-100.0]
        @test select(directional_problem(c -> c + 1, c -> 100c), 1.0, 3.0).flow == [100.0]
        @test select(directional_problem(c -> -c - 1, c -> 100c), 3.0, 1.0).flow == [100.0]
        inward = directional_problem(c -> c^2 - 2, c -> 100 * (c^2 - 2)^2)
        root = select(inward, 1.0, 2.0)
        @test root.drift == zeros(1, 1)
        @test abs(root.flow[1]) < 1e-20
        discrete = discretize(inward, FiniteDifference((3,)))
        coefficients = PolicyPDEs._pack_coefficients(discrete, fill(root, 3))
        system = PolicyPDEs._assemble_operator(discrete, coefficients; values = zeros(3, 1))
        @test all(iszero, system.coefficients.drift)
        @test iszero(norm(system.generator, Inf))
        @test select(directional_problem(c -> c - 2, c -> c), 2.0, 3.0).flow == [2.0]
        @test select(directional_problem(c -> c - 2, c -> c), 1.0, 2.0).flow == [2.0]
        discontinuous = directional_problem(c -> c < 2 ? -1.0 : 1.0, c -> 0)
        @test_throws ErrorException select(discontinuous, 1.0, 3.0)
        problem = HJBProblem(;
            discount = (x, v, Dv, D2v) -> [1.0],
            flow = (x, v, Dv, D2v) -> [sum(Dv)],
            drift = (x, v, Dv, D2v) -> Dv .- 2,
            diffusion = (x, v, Dv, D2v) -> zeros(2, 1),
            domain = RectangularDomain((0, 0), (1, 1)),
            boundary = test_boundary(2),
        )
        selected = PolicyPDEs._local_upwind_coefficients(
            problem,
            (0.5, 0.5),
            [0.0],
            fill(2.0, 2, 1),
            zeros(2, 1),
            reshape([3.0, 1.0], 2, 1),
            reshape([1.0, 3.0], 2, 1),
        )
        @test selected.drift == reshape([1.0, 0.0], 2, 1)
    end

    @testset "State-constraint data validation" begin
        boundary = RectangularBoundary(
            lower = StateConstraintBoundary(1),
            upper = StateConstraintBoundary(x -> [1.0]),
        )
        make_problem = (sigma, faces) -> HJBProblem(;
                discount = (args...) -> [1.0],
                flow = (args...) -> [1.0],
                drift = (x, v, Dv, D2v) -> Dv .- 1,
                diffusion = (args...) -> fill(sigma, 1, 1),
                domain = RectangularDomain(0, 1),
                boundary = faces,
            )
        @test_throws ArgumentError StateConstraintBoundary(Inf)
        @test_throws ArgumentError StateConstraintBoundary([NaN])
        problem = make_problem(0.0, boundary)
        discrete = discretize(problem, FiniteDifference((3,)))
        @test isempty(PolicyPDEs._boundary_rows(discrete, 1, spatial_operators(discrete)))
        @test_throws ArgumentError assemble_operator(
            discretize(make_problem(0.1, boundary), FiniteDifference((3,))),
            zeros(3, 1),
        )
        wrong = RectangularBoundary(
            lower = StateConstraintBoundary(2),
            upper = StateConstraintBoundary(1),
        )
        @test_throws ArgumentError assemble_operator(
            discretize(make_problem(0.0, wrong), FiniteDifference((3,))),
            zeros(3, 1),
        )
        mismatch = RectangularBoundary(
            lower = StateConstraintBoundary([1, 1]),
            upper = StateConstraintBoundary(1),
        )
        @test_throws DimensionMismatch assemble_operator(
            discretize(make_problem(0.0, mismatch), FiniteDifference((3,))),
            zeros(3, 1),
        )
    end

    @testset "Coupled diffusion with exact quadratic values" begin
        problem = quadratic_switching_problem()
        discrete = discretize(problem, FiniteDifference(([0.0, 0.05, 0.2, 0.55, 1.0],)))
        n = length(discrete.grid)
        system = assemble_operator(discrete, ones(n, 2))
        @test issparse(system.A)
        @test size(system.A) == (2n, 2n)
        @test norm(system.generator * ones(2n), Inf) < 1e-12
        @test length(system.boundary_rows) == 4
        solution = solve(discrete; initial_value = zeros(n, 2), atol = 1e-10, rtol = 0)
        expected = [component == 1 ? 1 + x^2 : 2 + 2x^2 for x in discrete.grid, component = 1:2]
        @test solution.converged
        @test solution.retcode === :Success
        @test solution.values ≈ expected atol = 1e-10
        @test solution.residual_norm < 1e-10
        @test solution.boundary_residual_norm < 1e-10
        @test solution.iterations == 1
    end

    @testset "Tensor Kronecker ordering and corner Dirichlet rows" begin
        exact = x -> [1 + x[1]^2 + x[2]^2 + x[1] * x[2]]
        face = LinearBoundary(0, 1, exact)
        problem = HJBProblem(;
            discount = (x, v, Dv, D2v) -> [1.0],
            flow = (x, v, Dv, D2v) -> exact(x) .- 0.13,
            drift = (x, v, Dv, D2v) -> zeros(2, 1),
            diffusion = (x, v, Dv, D2v) -> reshape([0.2, 0.3], 2, 1),
            domain = RectangularDomain([0.0, 0.0], [1.0, 2.0]),
            boundary = RectangularBoundary(lower = (face, face), upper = (face, face)),
        )
        discrete = discretize(problem, FiniteDifference((7, 9)))
        ops = spatial_operators(discrete)
        nodes = collect(discrete.grid)
        @test ops.central[1] * [x[1] for x in nodes] ≈ ones(length(nodes)) atol = 1e-12
        @test ops.central[2] * [x[2] for x in nodes] ≈ ones(length(nodes)) atol = 1e-12
        @test ops.second[1] * [x[1]^2 for x in nodes] ≈ fill(2.0, length(nodes)) atol = 1e-10
        @test ops.second[2] * [x[2]^2 for x in nodes] ≈ fill(2.0, length(nodes)) atol = 1e-10
        solution = solve(discrete; initial_value = x -> [0.0], atol = 1e-9, rtol = 0)
        @test solution.converged
        @test solution.values[:, 1] ≈ [exact(x)[1] for x in nodes] atol = 1e-9
        @test solution.boundary_residual_norm < 1e-9
    end

    @testset "Variable drift multiplies derivatives on the left" begin
        problem = HJBProblem(;
            discount = (x, v, Dv, D2v) -> [1.0],
            flow = (x, v, Dv, D2v) -> [0.0],
            drift = (x, v, Dv, D2v) -> reshape([1 + x], 1, 1),
            diffusion = (x, v, Dv, D2v) -> reshape([0.3], 1, 1),
            domain = RectangularDomain(0, 1),
            boundary = test_boundary(1),
        )
        discrete = discretize(problem, FiniteDifference((11,)))
        system = assemble_operator(discrete, zeros(11, 1))
        values = [x^2 for x in discrete.grid]
        applied = system.generator * values
        for node = 2:10
            x = discrete.grid[node]
            @test applied[node] ≈ (1 + x) * (2x + 0.1) + 0.09 atol = 1e-12
            for column = 1:11
                column == node && continue
                @test system.generator[node, column] >= 0
            end
        end
        @test norm(system.generator * ones(11), Inf) < 1e-12
    end

    @testset "Robin/Neumann row replacement" begin
        problem = HJBProblem(;
            discount = (x, v, Dv, D2v) -> [1.0],
            flow = (x, v, Dv, D2v) -> [0.5 + x],
            drift = (x, v, Dv, D2v) -> reshape([0.5], 1, 1),
            diffusion = (x, v, Dv, D2v) -> reshape([0.2], 1, 1),
            domain = RectangularDomain(0, 1),
            boundary = RectangularBoundary(
                lower = LinearBoundary(1, 1, 2),
                upper = LinearBoundary(1, 0, 1),
            ),
        )
        discrete = discretize(problem, FiniteDifference((15,)))
        solution = solve(discrete; initial_value = zeros(15, 1), atol = 1e-9, rtol = 0)
        @test solution.converged
        @test solution.values[:, 1] ≈ [1 + x for x in discrete.grid] atol = 1e-8
    end

    @testset "Active inequality versus slack HJB boundary rows" begin
        problem = HJBProblem(;
            discount = (x, v, Dv, D2v) -> [1.0],
            flow = (x, v, Dv, D2v) -> [1.0],
            drift = (x, v, Dv, D2v) -> zeros(1, 1),
            diffusion = (x, v, Dv, D2v) -> zeros(1, 1),
            domain = RectangularDomain(0, 1),
            boundary = RectangularBoundary(
                lower = LinearInequalityBoundary(0, -1, -2),
                upper = LinearInequalityBoundary(0, -1, 0),
            ),
        )
        discrete = discretize(problem, FiniteDifference((7,)))
        @test_throws DimensionMismatch assemble_operator(discrete, ones(6, 1))
        for initial in (zeros(7, 1), fill(4.0, 7, 1))
            solution = solve(discrete; initial_value = initial, atol = 1e-10, rtol = 0)
            @test solution.converged
            @test solution.values[:, 1] ≈ [2, 1, 1, 1, 1, 1, 1] atol = 1e-10
            @test count(solution.active_boundary) == 1
            @test solution.boundary_residual_norm < 1e-10
        end
        capped = HJBProblem(
            problem.discount,
            problem.drift,
            problem.diffusion,
            problem.flow;
            domain = problem.domain,
            boundary = RectangularBoundary(
                lower = LinearInequalityBoundary(0, 1, 0.5),
                upper = LinearInequalityBoundary(0, -1, 0),
            ),
        )
        cap_solution = solve(
            discretize(capped, FiniteDifference((7,)));
            initial_value = zeros(7, 1),
            atol = 1e-10,
            rtol = 0,
        )
        @test cap_solution.converged
        @test cap_solution.values[:, 1] ≈ [0.5, 1, 1, 1, 1, 1, 1] atol = 1e-10
    end

    @testset "Economic examples and consistent zero-drift policy updates" begin
        problem = two_income_problem()
        calibration = TwoIncomeExample.model
        for n in (31, 61)
            discrete = discretize(problem, FiniteDifference((n,)))
            solution = solve(
                discrete;
                dt = 100.0,
                initial_value = x ->
                    log.(calibration.income .+ calibration.interest * x) ./
                    calibration.discount,
                maxiters = 100,
            )
            @test solution.converged
            @test solution.iterations < 30
            @test solution.residual_norm < 1e-6
            @test solution.boundary_residual_norm < 1e-9
            @test minimum(diff(solution.values; dims = 1)) > -1e-8
            # State constraints keep the HJB at endpoints and do not impose
            # a derivative equation on the first two values.
            mu = solution.coefficients.drift[1, :, :]
            @test abs(mu[1, 1]) < 1e-10
            @test mu[1, 2] >= -1e-10
            @test mu[2, 1] < 0
            @test all(mu[end, :] .< 0)
            @test isempty(solution.linear_system.boundary_rows)
            @test norm(
                solution.linear_system.raw_A * vec(solution.values) -
                solution.linear_system.raw_b,
                Inf,
            ) < 1e-6
            ops = spatial_operators(discrete)
            backward, _ = PolicyPDEs._grid_derivatives(
                discrete,
                solution.values,
                ops,
                [];
                direction = :backward,
            )
            forward, _ = PolicyPDEs._grid_derivatives(
                discrete,
                solution.values,
                ops,
                [];
                direction = :forward,
            )
            @test isapprox(
                vec(backward[1, 2, :]),
                (solution.values[2, :] - solution.values[1, :]) / discrete.grid.steps[1][1],
            )
            @test isapprox(
                vec(backward[1, 1, :]),
                1 ./ (calibration.income .+ calibration.interest * discrete.grid[1]),
            )
            @test isapprox(
                vec(forward[1, end, :]),
                1 ./ (calibration.income .+ calibration.interest * discrete.grid[end]),
            )
        end
        discrete = discretize(coupled_diffusion_problem(), FiniteDifference((7, 7)))
        solution = solve(discrete; initial_value = zeros(49, 2), maxiters = 30)
        @test solution.converged
        @test solution.residual_norm < 1e-8
        @test solution.boundary_residual_norm < 1e-9
    end

    @testset "Nonlinear greedy policy and finite pseudo-time" begin
        # max_c [2c-(1+c^2)v] = 0 has v = 1 and greedy c = 1/v for v>0.
        problem = HJBProblem(;
            discount = (x, v, Dv, D2v) -> [1 + inv(v[1])^2],
            flow = (x, v, Dv, D2v) -> [2inv(v[1])],
            drift = (x, v, Dv, D2v) -> zeros(1, 1),
            diffusion = (x, v, Dv, D2v) -> zeros(1, 1),
            domain = RectangularDomain(0, 1),
            boundary = RectangularBoundary(
                lower = LinearBoundary(0, 1, 1),
                upper = LinearBoundary(0, 1, 1),
            ),
        )
        discrete = discretize(problem, FiniteDifference((9,)))
        for dt in (Inf, 0.5)
            solution = solve(
                discrete;
                dt,
                initial_value = fill(0.5, 9, 1),
                maxiters = 1000,
                atol = 1e-10,
                rtol = 0,
            )
            @test solution.converged
            @test solution.iterations > 1
            @test solution.values ≈ ones(9, 1) atol = 1e-9
            @test solution.coefficients.discount ≈ fill(2.0, 9, 1) atol = 1e-9
            @test solution.residual_norm < 1e-10
            @test length(solution.residual_history) == solution.iterations
        end
        default_step = solve(discrete; initial_value = fill(0.5, 9, 1), maxiters = 1)
        infinite_step = solve(discrete; dt = Inf, initial_value = fill(0.5, 9, 1), maxiters = 1)
        @test default_step.values == infinite_step.values
        @test default_step.residual_history == infinite_step.residual_history
        @test !isdefined(PolicyPDEs, :Picard)
        stopped = solve(discrete; initial_value = fill(0.5, 9, 1), maxiters = 1)
        @test !stopped.converged && stopped.retcode === :MaxIters
        @test stopped.residual_norm > 1e-3
        # Capture stdout to check the public trace and its default silence.
        for traced in (false, true)
            result, output = mktemp() do path, io
                result = redirect_stdout(io) do
                    options = traced ? (; show_trace = true) : (;)
                    solve(
                        discrete;
                        initial_value = fill(0.5, 9, 1),
                        maxiters = 1,
                        options...,
                    )
                end
                flush(io)
                seekstart(io)
                result, read(io, String)
            end
            @test result.values == stopped.values
            @test result.residual_history == stopped.residual_history
            if traced
                lines = split(chomp(output), '\n')
                @test length(lines) == 2
                @test lines[1] == lpad("Iteration", 9) * "  " * lpad("Residual", 24)
                @test lines[2] == lpad(1, 9) * "  " * lpad(result.residual_norm, 24)
            else
                @test isempty(output)
            end
        end
        @test_throws ArgumentError solve(
            discrete;
            initial_value = ones(9, 1),
            show_trace = 1,
        )
        @test_throws ArgumentError solve(discrete; initial_value = zeros(9, 1))
        @test_throws DimensionMismatch solve(discrete; initial_value = zeros(8, 1))
        @test_throws ArgumentError solve(discrete; initial_value = ones(9, 1), maxiters = 0)
    end

    @testset "Invalid options and conflicting corner data" begin
        discrete = discretize(quadratic_switching_problem(), FiniteDifference((3,)))
        for dt in (0, -1, NaN, -Inf, nextfloat(0.0))
            @test_throws ArgumentError solve(discrete; dt, initial_value = zeros(3, 2))
        end
        a = LinearBoundary(0, 1, 1)
        b = LinearBoundary(0, 1, 2)
        problem = HJBProblem(;
            discount = (x, v, Dv, D2v) -> [1.0],
            flow = (x, v, Dv, D2v) -> [1.0],
            drift = (x, v, Dv, D2v) -> zeros(2, 1),
            diffusion = (x, v, Dv, D2v) -> zeros(2, 1),
            domain = RectangularDomain([0, 0], [1, 1]),
            boundary = RectangularBoundary(lower = (a, b), upper = (a, b)),
        )
        @test_throws ArgumentError assemble_operator(
            discretize(problem, FiniteDifference((3, 3))),
            ones(9, 1),
        )
    end
end
