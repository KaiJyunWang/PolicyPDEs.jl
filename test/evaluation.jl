include(joinpath(@__DIR__,"..","examples","independent_diffusion_setup.jl"))

@testset "Bulk HJB coefficient and residual evaluation" begin
    @testset "Two-coordinate setup example" begin
        problem=independent_diffusion_problem()
        control=[0.5,0.5]
        coefficients=@inferred hjb_coefficients(problem,(1.0,1.0),control)
        @test coefficients.discount ≈ [0.056,0.056]
        @test coefficients.diffusion ≈ [0.05 0.05;0.1005 0.1005]
        @test hjb_residual(problem,(1.0,1.0),control,[1.0,2.0],zeros(2,2),zeros(2,2)) ≈
              [log(0.5)-0.0239,log(0.5)-0.12515]
    end

    @testset "Manufactured vector value: two coordinates, three regimes" begin
        problem=HJBProblem(;control=test_control,
            discount=(x,c)->[0.1+0.01*x[1]+0.02*c[i]+0.03*i for i in 1:3],
            drift=(x,c)->[k==1 ? x[1]+c[i] : i-x[2] for k in 1:2,i in 1:3],
            diffusion=(x,c)->[k==1 ? 0.2*x[1] : 0.1*(c[i]+i) for k in 1:2,i in 1:3],
            flow=(x,c)->[x[1]+2*x[2]+c[i]+i for i in 1:3],
            transition=(x,c)->[i==j ? 0.0 : 0.1*(x[1]+c[i]+i) for i in 1:3,j in 1:3],
            domain=RectangularDomain([0,0],[5,3]),boundary=test_boundary(2))
        x=(1.0,2.0)
        c=[0.5,0.5,0.5]
        v=x->[x[1]^2+3*x[2]^2+2*(i-1) for i in 1:3]
        gradient=[2.0 2.0 2.0;12.0 12.0 12.0]
        second=[2.0 2.0 2.0;6.0 6.0 6.0]
        coefficients=hjb_coefficients(problem,x,c)
        @test coefficients.discount ≈ [0.15,0.18,0.21]
        @test size(coefficients.drift) == (2,3)
        @test coefficients.transition ≈ [0 0.25 0.25;0.35 0 0.35;0.45 0.45 0]
        @test hjb_residual(problem,x,c,v(x),gradient,second) ≈ [-2.8425,8.0275,17.6375]
        @test_throws DimensionMismatch hjb_coefficients(problem,1.0,c)
        @test_throws DimensionMismatch hjb_residual(problem,x,c,[13.0],gradient,second)
        @test_throws DimensionMismatch hjb_residual(problem,x,c,v(x),transpose(gradient),second)
        @test_throws DimensionMismatch hjb_residual(problem,x,c,v(x),gradient,zeros(2,2))
        # Changing only source control 2 leaves other residual components intact.
        changed=hjb_residual(problem,x,[0.5,0.8,0.5],v(x),gradient,second)
        baseline=hjb_residual(problem,x,c,v(x),gradient,second)
        @test changed[1] == baseline[1]
        @test changed[2] != baseline[2]
        @test changed[3] == baseline[3]
    end

    @testset "Each primitive called once with the full control object" begin
        original=two_income_problem()
        calls=zeros(Int,5)
        controls_seen=[]
        names=(:discount,:drift,:diffusion,:flow,:transition)
        callbacks=map(enumerate(names)) do (k,name)
            (x,c)->begin
                calls[k]+=1
                push!(controls_seen,c)
                getproperty(original,name)(x,c)
            end
        end
        problem=HJBProblem(callbacks[1:4]...;control=test_control,domain=original.domain,
                           boundary=original.boundary,transition=callbacks[5])
        control=[1.0,1.0]
        @test hjb_residual(problem,1.0,control,[1.0,3.0],zeros(1,2),zeros(1,2)) ≈ [-0.01,-0.21]
        @test calls == ones(Int,5)
        @test all(c->c===control,controls_seen)
    end

    @testset "Single component, exact coefficients, and absent switching" begin
        callbacks=((x,c)->(1//2,), (x,c)->reshape([0],1,1),
                   (x,c)->reshape([0],1,1), (x,c)->[1])
        for transition in (nothing,(x,c)->zeros(1,1))
            problem=HJBProblem(callbacks...;control=test_control,domain=RectangularDomain(0,5),boundary=test_boundary(1),transition)
            coefficients=hjb_coefficients(problem,1,nothing)
            @test coefficients.discount === (1//2,)
            @test hjb_residual(problem,1,nothing,(2,),zeros(1,1),zeros(1,1)) == [0]
        end
        scalar_control=HJBProblem((x,c)->[0,0],(x,c)->zeros(1,2),(x,c)->zeros(1,2),
            (x,c)->[c,c^2];control=test_control,domain=RectangularDomain(0,5),boundary=test_boundary(1))
        @test hjb_residual(scalar_control,1,2,[0,0],zeros(1,2),zeros(1,2)) == [2,4]
    end

    @testset "Strict shapes and CTMC rate validation" begin
        make_problem(;discount=(x,c)->[0.05,0.05],drift=(x,c)->zeros(2,2),
                     diffusion=(x,c)->[0.2 0.2;0.3 0.3],flow=(x,c)->[1,1],
                     transition=nothing)=HJBProblem(discount,drift,diffusion,flow;
            control=test_control,domain=RectangularDomain([0,0],[5,3]),boundary=test_boundary(2),transition)
        x=[1.0,2.0]
        @test_throws ArgumentError hjb_coefficients(make_problem(diffusion=(x,c)->[0.2,0.3]),x,1)
        @test_throws DimensionMismatch hjb_coefficients(make_problem(diffusion=(x,c)->zeros(2,3)),x,1)
        @test_throws DimensionMismatch hjb_coefficients(make_problem(drift=(x,c)->zeros(1,2)),x,1)
        @test_throws ArgumentError hjb_coefficients(make_problem(discount=(x,c)->0.05),x,1)
        @test_throws ArgumentError hjb_coefficients(make_problem(discount=(x,c)->Float64[]),x,1)
        @test_throws DimensionMismatch hjb_coefficients(make_problem(discount=(x,c)->[0.05,0.05,0.05]),x,1)
        @test_throws DimensionMismatch hjb_coefficients(make_problem(flow=(x,c)->[1]),x,1)
        @test_throws ArgumentError hjb_coefficients(make_problem(discount=(x,c)->[Inf,1]),x,1)
        @test_throws ArgumentError hjb_coefficients(make_problem(diffusion=(x,c)->[NaN 0;0 0]),x,1)
        @test_throws ArgumentError hjb_coefficients(make_problem(),[1.0,Inf],1)
        @test_throws DimensionMismatch hjb_coefficients(make_problem(transition=(x,c)->zeros(3,3)),x,1)
        @test_throws ArgumentError hjb_coefficients(make_problem(transition=(x,c)->[0,1]),x,1)
        @test_throws ArgumentError hjb_coefficients(make_problem(transition=(x,c)->nothing),x,1)
        for rate in (-0.1,Inf,NaN,1+im)
            @test_throws ArgumentError hjb_coefficients(make_problem(transition=(x,c)->[0 rate;0.2 0]),x,1)
        end
        # A full generator is intentionally distinct from the off-diagonal rate matrix.
        @test_throws ArgumentError hjb_coefficients(make_problem(transition=(x,c)->[-0.1 0.1;0.2 -0.2]),x,1)
    end
end
