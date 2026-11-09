module TwoIncomeExample
# Keep the user's optional plotting section out of the numerical test environment.
example_path = joinpath(@__DIR__, "..", "examples", "solve_two_income.jl")
include_string(
    @__MODULE__,
    first(split(read(example_path, String), r"(?m)^using Plots[ \t]*\r?$"; limit = 2)),
    example_path,
)
end
two_income_problem() = TwoIncomeExample.problem

@testset "Direct-coefficient HJBProblem" begin
    discount = (x, v, Dv, D2v) -> [0.05]
    drift = (x, v, Dv, D2v) -> reshape([x - Dv[1, 1]], 1, 1)
    diffusion = (x, v, Dv, D2v) -> zeros(1, 1)
    flow = (x, v, Dv, D2v) -> [v[1]]
    domain = RectangularDomain(0, 5)
    boundary = test_boundary(1)
    problem = @inferred HJBProblem(discount, drift, diffusion, flow; domain, boundary)
    other = @inferred HJBProblem(; discount, drift, diffusion, flow, domain, boundary)
    @test typeof(problem) === typeof(other)
    @test !hasproperty(problem, :control)
    @test !isdefined(PolicyPDEs, :update_control)
    for name in (:discount, :drift, :diffusion, :flow, :domain, :boundary, :transition)
        @test getproperty(problem, name) === getproperty(other, name)
    end
    calls = Ref(0)
    bomb = (args...) -> (calls[] += 1; error("must remain lazy"))
    lazy = HJBProblem(bomb, bomb, bomb, bomb; domain, boundary, transition = bomb)
    @test calls[] == 0
    @test lazy.discount === bomb
    @test_throws UndefKeywordError HJBProblem(discount, drift, diffusion, flow; domain)
    @test_throws UndefKeywordError HJBProblem(; discount, drift, diffusion, flow, boundary)
    @test_throws MethodError HJBProblem(;
        discount,
        drift,
        diffusion,
        flow,
        domain,
        boundary,
        control = bomb,
    )
    @test_throws ArgumentError HJBProblem(;
        discount,
        drift,
        diffusion,
        flow,
        domain = (0, 5),
        boundary,
    )
    @test_throws ArgumentError HJBProblem(;
        discount,
        drift,
        diffusion,
        flow,
        domain,
        boundary = nothing,
    )
    @test_throws DimensionMismatch HJBProblem(;
        discount,
        drift,
        diffusion,
        flow,
        domain,
        boundary = test_boundary(2),
    )
    economic = two_income_problem()
    calibration = TwoIncomeExample.model
    @test economic.domain.lower == (calibration.asset_min,)
    @test economic.domain.upper == (calibration.asset_max,)
    coefficients = @inferred hjb_coefficients(economic, 1.0, [0.0, 0.0], [2.0 4.0], zeros(1, 2))
    @test coefficients.discount == fill(calibration.discount, 2)
    @test isapprox(coefficients.flow, log.([0.5, 0.25]))
    @test isapprox(
        coefficients.drift,
        reshape(calibration.income .+ calibration.interest .- [0.5, 0.25], 1, 2),
    )
    @test coefficients.transition == [0.0 calibration.low_to_high; calibration.high_to_low 0.0]
end
