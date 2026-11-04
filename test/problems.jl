include(joinpath(@__DIR__, "..", "examples", "two_income_setup.jl"))

@testset "HJBProblem" begin
    discount=(x,c)->[0.05]
    drift=(x,c)->reshape([1+0.03*x-c[1]],1,1)
    diffusion=(x,c)->zeros(1,1)
    flow=(x,c)->log.(c)
    control=test_control
    domain=RectangularDomain([0.0],[5.0])
    boundary=test_boundary(1)

    @testset "Constructor forms and preserved callback types" begin
        problem=@inferred HJBProblem(discount,drift,diffusion,flow;control,domain,boundary)
        other=@inferred HJBProblem(;discount,drift,diffusion,flow,control,domain,boundary)
        @test typeof(problem) === typeof(other)
        for name in (:discount,:drift,:diffusion,:flow,:control,:domain,:boundary,:transition)
            @test getproperty(problem,name) === getproperty(other,name)
        end
        @test problem.transition === nothing
        @test problem.drift(1.0,[0.5]) ≈ reshape([0.53],1,1)
    end

    @testset "No eager evaluation or copying" begin
        calls=Ref(0)
        component=(x,c)->(calls[]+=1; error("must not evaluate during setup"))
        lower=[0]
        stored_domain=RectangularDomain(lower,[5])
        stored_boundary=test_boundary(1)
        problem=HJBProblem(component,component,component,component;
            control=(v,Dv,D2v,x)->(calls[]+=1;error("control must remain lazy")),
            domain=stored_domain,boundary=stored_boundary,transition=component)
        @test calls[] == 0
        for name in (:discount,:drift,:diffusion,:flow,:transition)
            @test getproperty(problem,name) === component
        end
        @test problem.domain === stored_domain
        @test problem.boundary === stored_boundary
        lower[1]=-1
        @test problem.domain.lower == (0,)
    end

    @testset "Required inputs and geometry validation" begin
        @test_throws UndefKeywordError HJBProblem(discount,drift,diffusion,flow;domain,boundary)
        @test_throws UndefKeywordError HJBProblem(;discount,drift,diffusion,flow,domain,boundary)
        @test_throws ArgumentError HJBProblem(discount,drift,diffusion,flow;
                                              control,domain=(;asset=(0,5)),boundary)
        @test_throws ArgumentError HJBProblem(discount,drift,diffusion,flow;control,domain,boundary=nothing)
        @test_throws DimensionMismatch HJBProblem(discount,drift,diffusion,flow;
            control,domain=RectangularDomain([0,0],[5,3]),boundary)

        @test_throws MethodError HJBProblem(discount,drift,diffusion,flow;
                                            control,domain,boundary,regimes=(1,2))
        @test_throws UndefKeywordError HJBProblem(;control,domain,boundary)
        @test_throws UndefKeywordError HJBProblem(;discount,drift,diffusion,control,domain,boundary)
        @test_throws UndefKeywordError HJBProblem(;discount,drift,flow,control,domain,boundary)
        @test_throws UndefKeywordError HJBProblem(;discount,diffusion,flow,control,domain,boundary)
        @test_throws UndefKeywordError HJBProblem(;drift,diffusion,flow,control,domain,boundary)
        @test_throws UndefKeywordError HJBProblem(discount,drift,diffusion,flow;control,domain)
        @test_throws UndefKeywordError HJBProblem(discount,drift,diffusion,flow;boundary)
        @test_throws MethodError HJBProblem(discount,drift,diffusion,flow;control,domain,boundary,dt=Inf)
    end

    @testset "Callable structs: two-income setup on [0,5]" begin
        problem=two_income_problem()
        @test problem.domain.lower == (0.0,)
        @test problem.domain.upper == (5.0,)
        @test problem.discount(1.0,[0.1,0.2]) == [0.05,0.05]
        @test problem.drift(1.0,[0.1,0.2]) ≈ [0.03 0.03]
        @test problem.diffusion(1.0,[0.1,0.2]) == zeros(1,2)
        @test problem.flow(1.0,[0.1,0.2]) ≈ log.([0.1,0.2])
        @test problem.transition(1.0,[0.1,0.2]) == [0.0 0.02;0.03 0.0]
        lower=problem.boundary.lower[1]
        upper=problem.boundary.upper[1]
        @test lower isa LinearInequalityBoundary
        @test upper isa LinearEqualityBoundary
        marginal=[10.0,5.0]
        @test boundary_residual(lower,[0,0],marginal;x=0.0) ≈ [0,0]
        @test boundary_residual(lower,[0,0],marginal .+ 1;x=0.0) ≈ [-1,-1]
        @test boundary_residual(lower,[0,0],marginal .- 1;x=0.0) ≈ [1,1]
        @test boundary_residual(upper,[0,0],inv.([0.1,0.2] .+ 0.03*5);x=5.0) ≈ [0,0]
    end
end
