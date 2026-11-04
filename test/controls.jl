struct DerivativeControl end
(::DerivativeControl)(v,Dv,D2v,x) =
    [v[1]+Dv[1,1]+D2v[2,1]+x[1],v[2]+Dv[2,2]+D2v[1,2]+x[2]]

function control_test_problem(rule)
    bomb=(args...)->error("control evaluation must not call HJB primitives")
    return HJBProblem(bomb,bomb,bomb,bomb;control=rule,
        domain=RectangularDomain([0,0],[5,5]),boundary=test_boundary(2))
end

@testset "User-supplied control update" begin
    values=[3.0,4.0]
    gradient=[5.0 6.0;7.0 8.0]
    second=[9.0 10.0;11.0 12.0]
    x=(1.0,2.0)

    @testset "Callable struct uses values, both derivatives, and state" begin
        problem=control_test_problem(DerivativeControl())
        @test update_control(problem,x,values,gradient,second) == [20.0,24.0]
    end

    @testset "Lazy construction, argument order, and returned identity" begin
        calls=Ref(0)
        seen=Ref{Any}(nothing)
        result=[1//2,2//3,3//4] # Model-defined control length need not equal m.
        rule=(v,Dv,D2v,state)->begin
            calls[]+=1
            seen[]=(v,Dv,D2v,state)
            result
        end
        problem=control_test_problem(rule)
        @test problem.control === rule
        @test calls[] == 0
        @test update_control(problem,x,values,gradient,second) === result
        @test calls[] == 1
        @test seen[][1] === values
        @test seen[][2] === gradient
        @test seen[][3] === second
        @test seen[][4] === x
        @test eltype(result) === Rational{Int}
    end

    @testset "Input and output validation" begin
        calls=Ref(0)
        problem=control_test_problem((args...)->(calls[]+=1;[1.0]))
        @test_throws DomainError update_control(problem,(-1.0,2.0),values,gradient,second)
        @test_throws DimensionMismatch update_control(problem,1.0,values,gradient,second)
        @test_throws ArgumentError update_control(problem,x,Float64[],zeros(2,0),zeros(2,0))
        @test_throws ArgumentError update_control(problem,x,reshape(values,1,2),gradient,second)
        @test_throws ArgumentError update_control(problem,x,[Inf,1],gradient,second)
        @test_throws DimensionMismatch update_control(problem,x,values,zeros(2,1),second)
        @test_throws ArgumentError update_control(problem,x,values,gradient,zeros(2,2,2))
        @test calls[] == 0
        for invalid in (1.0,(1.0,2.0),[1.0 2.0],[Inf],[NaN],[1+im],[[1.0]])
            bad=control_test_problem((v,Dv,D2v,x)->invalid)
            @test_throws ArgumentError update_control(bad,x,values,gradient,second)
        end
    end

    @testset "Residual overload evaluates policy and coefficients once" begin
        base=two_income_problem()
        counts=zeros(Int,6)
        returned=[1.0,1.0]
        seen=[]
        names=(:discount,:drift,:diffusion,:flow,:transition)
        callbacks=map(enumerate(names)) do (k,name)
            (x,c)->begin
                counts[k]+=1
                push!(seen,c)
                getproperty(base,name)(x,c)
            end
        end
        rule=(v,Dv,D2v,x)->(counts[6]+=1;returned)
        problem=HJBProblem(callbacks[1:4]...;control=rule,domain=base.domain,
                           boundary=base.boundary,transition=callbacks[5])
        @test hjb_residual(problem,1.0,[1.0,3.0],zeros(1,2),zeros(1,2)) ≈ [-0.01,-0.21]
        @test counts == ones(Int,6)
        @test all(c->c===returned,seen)
    end

    @testset "Log consumption rule and constrained endpoint controls" begin
        problem=two_income_problem()
        @test update_control(problem,1.0,[0,0],[2.0 4.0],zeros(1,2)) ≈ [0.5,0.25]
        @test update_control(problem,0.0,[0,0],[2.0 4.0],zeros(1,2)) ≈ [0.1,0.2]
        @test update_control(problem,5.0,[0,0],[10.0 10.0],zeros(1,2)) ≈ [0.25,0.35]
        @test_throws DomainError update_control(problem,1.0,[0,0],zeros(1,2),zeros(1,2))
    end

    @testset "Two-coordinate supplied maximizer agrees with grid residuals" begin
        problem=independent_diffusion_problem()
        v=[1.0,2.0]
        Dv=[2.0 3.0;-1.0 2.0]
        D2v=[-1.0 -2.0;-3.0 -4.0]
        point=(1.0,1.0)
        candidates=collect(0.1:0.1:1.0)
        residuals=[hjb_residual(problem,point,fill(c,2),v,Dv,D2v) for c in candidates]
        chosen=update_control(problem,point,v,Dv,D2v)
        for i in 1:2
            @test chosen[i] == candidates[argmax([r[i] for r in residuals])]
        end
        @test hjb_residual(problem,point,v,Dv,D2v) ≈
              [maximum(r[i] for r in residuals) for i in 1:2]
    end
end
