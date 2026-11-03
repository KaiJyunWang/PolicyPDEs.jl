struct StateConstraints end
include(joinpath(@__DIR__, "..", "examples", "two_income_setup.jl"))

@testset "HJBProblem" begin
    discount = (x, a, i) -> 0.05
    drift = (x, a, i) -> 1 + 0.03 * x - a
    diffusion = (x, a, i) -> 0.0
    flow = (x, a, i) -> log(a)
    domain = (; assets = (0.0, 5.0))
    boundary = StateConstraints()

    @testset "Single equation and constructor forms" begin
        problem = @inferred HJBProblem(discount, drift, diffusion, flow; domain, boundary)
        keyword_problem = @inferred HJBProblem(; discount, drift, diffusion, flow,
                                              domain, boundary)
        @test typeof(problem) === typeof(keyword_problem)
        for name in (:discount, :drift, :diffusion, :flow, :domain, :boundary,
                     :regimes, :transition)
            @test getproperty(problem, name) === getproperty(keyword_problem, name)
        end
        @test problem.regimes === (1,)
        @test problem.transition === nothing
        @test problem.drift(1.0, 0.5, 1) ≈ 0.53
        @test problem.flow(1.0, 1.0, 1) == 0.0
    end

    @testset "No eager evaluation or copying" begin
        calls = Ref(0)
        component = (args...) -> (calls[] += 1; error("must not evaluate during setup"))
        mutable_domain = Dict(:assets => (0, 100))
        mutable_boundary = Dict(:lower => :state_constraint)
        labels = [:low, :middle, :high]
        problem = HJBProblem(component, component, component, component;
                             domain = mutable_domain, boundary = mutable_boundary,
                             regimes = labels, transition = component)
        @test calls[] == 0
        @test problem.discount === component
        @test problem.drift === component
        @test problem.diffusion === component
        @test problem.flow === component
        @test problem.transition === component
        @test problem.domain === mutable_domain
        @test problem.boundary === mutable_boundary
        @test problem.regimes === labels
        mutable_domain[:assets] = (0, 200)
        @test problem.domain[:assets] == (0, 200)
    end

    @testset "Three coupled equations with state and control dependent rates" begin
        labels = (:low, :middle, :high)
        rate = (x, a, i, j) -> begin
            i == j && error("only off-diagonal rates are supplied")
            (i == :low ? 0.1 : 0.2) * (j == :high ? 2 : 1) * (1 + x + a)
        end
        problem = @inferred HJBProblem(discount, drift, diffusion, flow;
                                      domain, boundary, regimes = labels, transition = rate)
        values = Dict(:low => 1.0, :middle => 2.0, :high => 4.0)
        coupling = sum(problem.transition(1.0, 0.5, :low, j) *
                       (values[j] - values[:low]) for j in labels if j != :low)
        @test coupling ≈ 1.75
        @test problem.transition(2.0, 0.5, :low, :high) ≈ 0.7
        @test problem.transition(1.0, 1.5, :low, :high) ≈ 0.7
        @test sum(problem.transition(1.0, 0.5, :low, j) * (3.0 - 3.0)
                  for j in labels if j != :low) == 0
    end

    @testset "Uncoupled systems and regime validation" begin
        problem = HJBProblem(discount, drift, diffusion, flow;
                             domain, boundary, regimes = 1:3)
        @test problem.regimes == 1:3
        @test problem.transition === nothing
        for invalid in ((), Int[], (1, 1), [:low, :low], 2, Set([1, 2]))
            @test_throws ArgumentError HJBProblem(discount, drift, diffusion, flow;
                                                 domain, boundary, regimes = invalid)
        end
        @test_throws ArgumentError HJBProblem(discount, drift, diffusion, flow,
                                             domain, boundary, (), nothing)
    end

    @testset "Required components and separation from solver settings" begin
        @test_throws UndefKeywordError HJBProblem(; domain, boundary)
        @test_throws UndefKeywordError HJBProblem(; discount, drift, diffusion, domain, boundary)
        @test_throws UndefKeywordError HJBProblem(; discount, drift, flow, domain, boundary)
        @test_throws UndefKeywordError HJBProblem(; discount, diffusion, flow, domain, boundary)
        @test_throws UndefKeywordError HJBProblem(; drift, diffusion, flow, domain, boundary)
        @test_throws UndefKeywordError HJBProblem(discount, drift, diffusion, flow; domain)
        @test_throws UndefKeywordError HJBProblem(discount, drift, diffusion, flow; boundary)
        @test_throws MethodError HJBProblem(discount, drift, diffusion, flow;
                                            domain, boundary, dt = Inf)
    end

    @testset "Callable structs: two-income switching example" begin
        problem = @inferred two_income_problem()
        @test problem.regimes == (1, 2)
        @test problem.domain.assets == (0.0, 5.0)
        @test problem.boundary.assets.lower isa LinearInequalityBoundary
        @test problem.boundary.assets.upper isa LinearEqualityBoundary
        lower = problem.boundary.assets.lower
        upper = problem.boundary.assets.upper
        for (regime, income) in ((1, 0.1), (2, 0.2))
            @test boundary_residual(lower, 0.0, inv(income); x = 0.0, regime) ≈ 0.0
            @test boundary_residual(lower, 0.0, inv(income) + 1; x = 0.0, regime) < 0
            @test boundary_residual(lower, 0.0, inv(income) - 1; x = 0.0, regime) > 0
            @test boundary_residual(upper, 0.0, inv(income + 0.03 * 5.0);
                                    x = 5.0, regime) ≈ 0.0
        end
        @test problem.discount(1.0, 0.1, 1) == 0.05
        @test problem.drift(1.0, 0.1, 1) ≈ 0.03
        @test problem.drift(1.0, 0.1, 2) ≈ 0.13
        @test problem.diffusion(1.0, 0.1, 1) == 0.0
        @test problem.flow(1.0, 0.1, 1) ≈ log(0.1)
        @test problem.transition(1.0, 0.1, 1, 2) == 0.02
        @test problem.transition(1.0, 0.1, 2, 1) == 0.03
        @test problem.transition(1.0, 0.1, 1, 2) * (3 - 1) ≈ 0.04
        @test problem.transition(1.0, 0.1, 2, 1) * (1 - 3) ≈ -0.06
    end

    @testset "Volatility convention and numeric types" begin
        rational_discount = (x, a, i) -> 1 // 20
        scalar_volatility = (x, a, i) -> 0.2
        problem = HJBProblem(rational_discount, drift, scalar_volatility, flow;
                             domain, boundary)
        @test problem.discount(1, 1, 1) === 1 // 20
        @test problem.diffusion(1, 1, 1)^2 / 2 ≈ 0.02

        vector_drift = (x, a, i) -> [a, -a]
        matrix_volatility = (x, a, i) -> [0.2 0.0; 0.1 0.3]
        vector_problem = HJBProblem(discount, vector_drift, matrix_volatility, flow;
                                    domain = (; x = (0, 1), y = (0, 1)), boundary)
        @test vector_problem.drift([0.5, 0.5], 0.1, 1) == [0.1, -0.1]
        sigma = vector_problem.diffusion([0.5, 0.5], 0.1, 1)
        @test sigma * sigma' / 2 ≈ [0.02 0.01; 0.01 0.05]
    end
end
