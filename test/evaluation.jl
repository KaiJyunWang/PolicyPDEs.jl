# Private manufactured fixture; numerical coverage does not depend on examples.
function coupled_diffusion_problem()
    HJBProblem(;
        discount = (x, v, Dv, D2v) -> [1.0, 1.0],
        flow = (x, v, Dv, D2v) -> [1.0, 2.0],
        drift = (x, v, Dv, D2v) -> zeros(2, 2),
        diffusion = (x, v, Dv, D2v) -> [0.1 0.2; 0.3 0.4],
        transition = (x, v, Dv, D2v) -> [0.0 0.02; 0.03 0.0],
        domain = RectangularDomain((0.0, 0.5), (5.0, 1.5)),
        boundary = test_boundary(2),
    )
end

@testset "Direct coefficient and residual evaluation" begin
    calls = zeros(Int, 5)
    seen = []
    outputs = ([0.5, 0.6], [1.0 2.0; 3.0 4.0], [0.2 0.3; 0.4 0.5], [5.0, 6.0], [0.0 0.2; 0.3 0.0])
    callbacks = map(1:5) do k
        (x, v, Dv, D2v) -> begin
            calls[k] += 1
            push!(seen, (x, v, Dv, D2v))
            outputs[k]
        end
    end
    problem = HJBProblem(
        callbacks[1:4]...;
        transition = callbacks[5],
        domain = RectangularDomain((0, 0), (5, 5)),
        boundary = test_boundary(2),
    )
    x = (1.0, 2.0)
    v = [2.0, 3.0]
    Dv = [1.0 2.0; 3.0 4.0]
    D2v = [-1.0 -2.0; -3.0 -4.0]
    coefficients = hjb_coefficients(problem, x, v, Dv, D2v)
    @test calls == ones(Int, 5)
    @test all(t -> t[1] === x && t[2] === v && t[3] === Dv && t[4] === D2v, seen)
    for (name, output) in zip((:discount, :drift, :diffusion, :flow, :transition), outputs)
        @test getproperty(coefficients, name) === output
    end
    fill!(calls, 0)
    residual = hjb_residual(problem, x, v, Dv, D2v)
    expected = [
        outputs[4][i] - outputs[1][i] * v[i] +
        sum(outputs[2][k, i] * Dv[k, i] + outputs[3][k, i]^2 * D2v[k, i] / 2 for k = 1:2) +
        sum(outputs[5][i, j] * (v[j] - v[i]) for j = 1:2) for i = 1:2
    ]
    @test isapprox(residual, expected)
    @test calls == ones(Int, 5)
    @test_throws DimensionMismatch hjb_coefficients(problem, x, v, zeros(1, 2), D2v)
    @test_throws DimensionMismatch hjb_coefficients(problem, x, v, Dv, zeros(2, 1))
    @test_throws DomainError hjb_coefficients(problem, (6.0, 1.0), v, Dv, D2v)
    @test_throws ArgumentError hjb_coefficients(problem, x, [NaN, 1.0], Dv, D2v)
    @test_throws ArgumentError hjb_coefficients(
        problem,
        x,
        Float64[],
        zeros(2, 0),
        zeros(2, 0),
    )
    @test_throws MethodError hjb_coefficients(problem, x, [1.0, 1.0])
    make_problem(;
        discount = (x, v, Dv, D2v) -> [1.0],
        drift = (x, v, Dv, D2v) -> zeros(1, 1),
        diffusion = (x, v, Dv, D2v) -> zeros(1, 1),
        flow = (x, v, Dv, D2v) -> [1.0],
        transition = nothing,
    ) = HJBProblem(;
        discount,
        drift,
        diffusion,
        flow,
        transition,
        domain = RectangularDomain(0, 1),
        boundary = test_boundary(1),
    )
    evaluate = p -> hjb_coefficients(p, 0.5, [1.0], zeros(1, 1), zeros(1, 1))
    @test evaluate(make_problem()).transition === nothing
    @test_throws ArgumentError evaluate(make_problem(discount = (args...) -> 1.0))
    @test_throws DimensionMismatch evaluate(make_problem(flow = (args...) -> [1.0, 2.0]))
    @test_throws ArgumentError evaluate(make_problem(drift = (args...) -> [0.0]))
    @test_throws ArgumentError evaluate(
        make_problem(diffusion = (args...) -> fill(Inf, 1, 1)),
    )
    @test_throws ArgumentError evaluate(make_problem(transition = (args...) -> ones(1, 1)))
    rates = HJBProblem(;
        discount = (args...) -> [1.0, 1.0],
        flow = (args...) -> [0.0, 0.0],
        drift = (args...) -> zeros(1, 2),
        diffusion = (args...) -> zeros(1, 2),
        transition = (args...) -> [0.0 -1.0; 0.0 0.0],
        domain = RectangularDomain(0, 1),
        boundary = test_boundary(1),
    )
    @test_throws ArgumentError hjb_coefficients(
        rates,
        0.5,
        [1.0, 2.0],
        zeros(1, 2),
        zeros(1, 2),
    )
end
