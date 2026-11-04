include(joinpath(@__DIR__, "..", "examples", "independent_diffusion_setup.jl"))

@testset "Independent diffusion and CTMC evaluation" begin
    @testset "Runnable independent-coordinate setup example" begin
        problem = @inferred independent_diffusion_problem()
        coefficients = hjb_coefficients(problem, (1.0,1.0), 0.5, 1)
        @test coefficients.discount ≈ 0.056
        @test all(isapprox.(coefficients.diffusion, (0.05,0.1005)))
        @test hjb_residual(problem, (1.0,1.0), 0.5, (1.0,2.0), (0.0,0.0),
                           (0.0,0.0); regime=1) ≈ log(0.5) - 0.0239
    end
    @testset "Two-dimensional manufactured vector value" begin
        # v_i(x) = x_1^2 + 3*x_2^2 + 2*(i-1), so mixed derivatives vanish.
        problem = HJBProblem(
            (x, c, i) -> 0.1 + 0.01*x[1] + 0.02*c + 0.03*i,
            (x, c, i) -> (x[1] + c, i - x[2]),
            (x, c, i) -> (0.2*x[1], 0.1*(c+i)),
            (x, c, i) -> x[1] + 2*x[2] + c + i;
            domain = (; asset = (0.0, 5.0), productivity = (0.0, 3.0)),
            boundary = nothing,
            regimes = (1, 2),
            transition = (x, c, i, j) -> begin
                i == j && error("diagonal intensities must not be requested")
                0.1 * (x[1] + c + i)
            end,
        )
        x = (1.0, 2.0)
        v = x -> [x[1]^2 + 3*x[2]^2, x[1]^2 + 3*x[2]^2 + 2]
        coefficients = @inferred hjb_coefficients(problem, x, 0.5, 1)
        @test coefficients.discount ≈ 0.15
        @test coefficients.drift == (1.5, -1.0)
        @test all(isapprox.(coefficients.diffusion, (0.2, 0.15)))
        @test coefficients.flow == 6.5
        # flow 6.5 - discount*value 1.95 + drift -9 + diffusion .1075 + jumps .5.
        @test hjb_residual(problem, x, 0.5, v(x), (2.0, 12.0), (2.0, 6.0);
                           regime = 1) ≈ -3.8425
        # Same continuous point and derivatives, but the second value component
        # has a different discount, drift, and outgoing switching rate.
        @test hjb_residual(problem, x, 0.5, v(x), (2.0, 12.0), (2.0, 6.0);
                           regime = 2) ≈ 7.3275
        @test_throws DimensionMismatch hjb_coefficients(problem, 1.0, 0.5)
        @test_throws DimensionMismatch hjb_residual(problem, x, 0.5, [13.0], (2, 12), (2, 6))
        @test_throws DimensionMismatch hjb_residual(problem, x, 0.5, v(x), (2,), (2, 6))
        @test_throws ArgumentError hjb_residual(problem, x, 0.5, v(x), (2, 12), [2 0; 0 6])
    end

    @testset "Scalar state, component ordering, and source control" begin
        problem = two_income_problem()
        @test hjb_residual(problem, 1.0, 1.0, [1.0, 3.0], 0.0, 0.0; regime=1) ≈ -0.01
        @test hjb_residual(problem, 1.0, 1.0, [1.0, 3.0], 0.0, 0.0; regime=2) ≈ -0.21
        labels = (:high, :low)
        controls_seen = []
        labeled = HJBProblem((x,c,i)->0, (x,c,i)->0, (x,c,i)->0, (x,c,i)->0;
            domain = (; asset=(0,5)), boundary=nothing, regimes=labels,
            transition = (x, c, i, j) -> begin
                push!(controls_seen, (c, i, j))
                c.rate
            end)
        control = (; rate=0.2)
        @test hjb_residual(labeled, 1, control, [5, 2], 0, 0; regime=:high) ≈ -0.6
        @test controls_seen == [(control, :high, :low)]
        @test_throws ArgumentError hjb_coefficients(labeled, 1, control, :missing)
    end

    @testset "Single equation and no switching" begin
        callbacks = ((x,c,i)->1//2, (x,c,i)->0, (x,c,i)->0, (x,c,i)->1)
        for transition in (nothing, (args...)->error("single component has no off-diagonal rate"))
            problem = HJBProblem(callbacks...; domain=(; asset=(0,5)), boundary=nothing,
                                 transition)
            coefficients = hjb_coefficients(problem, 1, nothing)
            @test coefficients.discount === 1//2
            @test hjb_residual(problem, 1, nothing, (2,), 0, 0) == 0
        end
        uncoupled = HJBProblem(callbacks...; domain=(; asset=(0,5)), boundary=nothing,
                               regimes=(1,2))
        @test hjb_residual(uncoupled, (1,), nothing, (2,4), (0,), (0,); regime=2) == -1
    end

    @testset "Coefficient shapes, finite outputs, and CTMC rates" begin
        make_problem(; discount=(x,c,i)->0.05, drift=(x,c,i)->(0.0,0.0),
                     diffusion=(x,c,i)->(0.2,0.3), flow=(x,c,i)->1.0,
                     transition=nothing) = HJBProblem(discount, drift, diffusion, flow;
            domain=(; asset=(0,5), productivity=(0,3)), boundary=nothing,
            regimes=(1,2), transition)
        x = [1.0, 2.0]
        @test_throws ArgumentError hjb_coefficients(make_problem(diffusion=(x,c,i)->[0.2 0; 0 0.3]), x, 1)
        @test_throws ArgumentError hjb_coefficients(make_problem(diffusion=(x,c,i)->0.2), x, 1)
        @test_throws DimensionMismatch hjb_coefficients(make_problem(drift=(x,c,i)->(1,)), x, 1)
        @test_throws ArgumentError hjb_coefficients(make_problem(discount=(x,c,i)->Inf), x, 1)
        @test_throws ArgumentError hjb_coefficients(make_problem(flow=(x,c,i)->[1,2]), x, 1)
        @test_throws ArgumentError hjb_coefficients(make_problem(diffusion=(x,c,i)->(NaN,0.3)), x, 1)
        @test_throws ArgumentError hjb_coefficients(make_problem(), [1.0, Inf], 1)
        for rate in (-0.1, Inf, NaN, 1+im)
            invalid = make_problem(transition=(x,c,i,j)->rate)
            @test_throws ArgumentError hjb_residual(invalid, x, 1, (1,2), (0,0), (0,0))
        end
    end
end
