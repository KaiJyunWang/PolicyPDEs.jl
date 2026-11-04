struct StateConstraints end
include(joinpath(@__DIR__, "..", "examples", "two_income_setup.jl"))

@testset "HJBProblem" begin
    discount=(x,c)->[0.05]
    drift=(x,c)->reshape([1+0.03*x-c[1]],1,1)
    diffusion=(x,c)->zeros(1,1)
    flow=(x,c)->log.(c)
    domain=(; assets=(0.0,5.0))
    boundary=StateConstraints()

    @testset "Constructor forms and preserved callback types" begin
        problem=@inferred HJBProblem(discount,drift,diffusion,flow;domain,boundary)
        other=@inferred HJBProblem(;discount,drift,diffusion,flow,domain,boundary)
        @test typeof(problem) === typeof(other)
        for name in (:discount,:drift,:diffusion,:flow,:domain,:boundary,:regimes,:transition)
            @test getproperty(problem,name) === getproperty(other,name)
        end
        @test problem.regimes == (1,)
        @test problem.transition === nothing
        @test problem.drift(1.0,[0.5]) ≈ reshape([0.53],1,1)
    end

    @testset "No eager evaluation or copying" begin
        calls=Ref(0)
        component=(x,c)->(calls[]+=1; error("must not evaluate during setup"))
        mutable_domain=Dict(:assets=>(0,5))
        mutable_boundary=Dict(:lower=>:state_constraint)
        labels=[:low,:middle,:high]
        problem=HJBProblem(component,component,component,component;
            domain=mutable_domain,boundary=mutable_boundary,regimes=labels,transition=component)
        @test calls[] == 0
        for name in (:discount,:drift,:diffusion,:flow,:transition)
            @test getproperty(problem,name) === component
        end
        @test problem.domain === mutable_domain
        @test problem.boundary === mutable_boundary
        @test problem.regimes === labels
        mutable_domain[:assets]=(0,10)
        @test problem.domain[:assets] == (0,10)
    end

    @testset "Regime metadata and required inputs" begin
        problem=HJBProblem(discount,drift,diffusion,flow;domain,boundary,regimes=1:3)
        @test problem.regimes == 1:3
        for invalid in ((),Int[],(1,1),[:low,:low],2,Set([1,2]))
            @test_throws ArgumentError HJBProblem(discount,drift,diffusion,flow;
                                                domain,boundary,regimes=invalid)
        end
        @test_throws ArgumentError HJBProblem(discount,drift,diffusion,flow,
                                             domain,boundary,(),nothing)
        @test_throws UndefKeywordError HJBProblem(;domain,boundary)
        @test_throws UndefKeywordError HJBProblem(;discount,drift,diffusion,domain,boundary)
        @test_throws UndefKeywordError HJBProblem(;discount,drift,flow,domain,boundary)
        @test_throws UndefKeywordError HJBProblem(;discount,diffusion,flow,domain,boundary)
        @test_throws UndefKeywordError HJBProblem(;drift,diffusion,flow,domain,boundary)
        @test_throws UndefKeywordError HJBProblem(discount,drift,diffusion,flow;domain)
        @test_throws UndefKeywordError HJBProblem(discount,drift,diffusion,flow;boundary)
        @test_throws MethodError HJBProblem(discount,drift,diffusion,flow;domain,boundary,dt=Inf)
    end

    @testset "Callable structs: two-income setup on [0,5]" begin
        problem=@inferred two_income_problem()
        @test problem.domain.assets == (0.0,5.0)
        @test problem.discount(1.0,[0.1,0.2]) == [0.05,0.05]
        @test problem.drift(1.0,[0.1,0.2]) ≈ [0.03 0.03]
        @test problem.diffusion(1.0,[0.1,0.2]) == zeros(1,2)
        @test problem.flow(1.0,[0.1,0.2]) ≈ log.([0.1,0.2])
        @test problem.transition(1.0,[0.1,0.2]) == [0.0 0.02;0.03 0.0]
        lower,upper=problem.boundary.assets
        @test lower isa LinearInequalityBoundary
        @test upper isa LinearEqualityBoundary
        for (regime,income) in ((1,0.1),(2,0.2))
            @test boundary_residual(lower,0.0,inv(income);x=0.0,regime) ≈ 0.0
            @test boundary_residual(lower,0.0,inv(income)+1;x=0.0,regime) < 0
            @test boundary_residual(lower,0.0,inv(income)-1;x=0.0,regime) > 0
            @test boundary_residual(upper,0.0,inv(income+0.03*5);x=5.0,regime) ≈ 0.0
        end
    end
end
