struct SavingsModel{T}
    discount::T
    interest::T
    income::T
end

struct StateConstraints end

@testset "HJBProblem" begin
    @testset "User-defined model and explicit problem data" begin
        model = SavingsModel(0.05, 0.03, 1.0)
        domain = (; assets = (0.0, 100.0))
        boundary = StateConstraints()
        problem = @inferred HJBProblem(model; domain, boundary)

        @test problem.model === model
        @test problem.domain === domain
        @test problem.boundary === boundary
        @test typeof(problem) === HJBProblem{typeof(model),typeof(domain),typeof(boundary)}
    end

    @testset "No copying, conversion, or eager model evaluation" begin
        model = (; discount = 1 // 20,
                 reward = (x, a) -> error("Construction must not evaluate reward"))
        domain = Dict(:assets => (0, 100))
        boundary = Dict(:lower => :state_constraint)
        problem = HJBProblem(model; domain, boundary)

        @test problem.model === model
        @test problem.model.discount isa Rational{Int}
        @test problem.domain === domain
        @test problem.boundary === boundary
        domain[:assets] = (0, 200)
        @test problem.domain[:assets] == (0, 200)
    end

    @testset "Required data and stage separation" begin
        model = SavingsModel(0.05, 0.03, 1.0)
        domain = (; assets = (0.0, 100.0))
        boundary = StateConstraints()

        @test_throws UndefKeywordError HJBProblem(model)
        @test_throws UndefKeywordError HJBProblem(model; domain)
        @test_throws UndefKeywordError HJBProblem(model; boundary)
        @test_throws MethodError HJBProblem(model; domain, boundary, dt = Inf)
    end
end
