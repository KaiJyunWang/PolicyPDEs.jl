@testset "Derivative-dependent coefficients and frozen snapshots" begin
    problem = two_income_problem()
    calibration = TwoIncomeExample.model
    # Coefficients no longer clamp consumption based solely on endpoint location.
    @test isapprox(
        hjb_coefficients(problem, 0.0, [0, 0], [2.0 4.0], zeros(1, 2)).drift,
        reshape(calibration.income .- [0.5, 0.25], 1, 2),
    )
    @test isapprox(
        hjb_coefficients(problem, 5.0, [0, 0], [10.0 10.0], zeros(1, 2)).drift,
        reshape(calibration.income .+ calibration.interest * 5.0 .- 0.1, 1, 2),
    )
    @test_throws DomainError hjb_coefficients(
        problem,
        1.0,
        [0, 0],
        zeros(1, 2),
        zeros(1, 2),
    )
    discrete = discretize(problem, FiniteDifference((11,)))
    values = [
        log(y + calibration.interest * x) / calibration.discount for x in discrete.grid,
        y in calibration.income
    ]
    system = assemble_operator(discrete, values)
    @test size(system.coefficients.flow) == (11, 2)
    @test size(system.coefficients.drift) == (1, 11, 2)
    @test size(system.coefficients.transition) == (11, 2, 2)
    saved = copy(system.coefficients.flow)
    values .+= 1
    @test system.coefficients.flow == saved
    @test !hasproperty(system, :controls)
end
