struct VectorBoundaryRHS{T}
    base::T
end
(rhs::VectorBoundaryRHS)(x) = rhs.base .+ x .+ [1,2]

@testset "Linear face boundaries" begin
    @testset "Equality and inequality residuals" begin
        dirichlet=@inferred LinearEqualityBoundary(a=0,b=1,c=2)
        @test boundary_residual(dirichlet,[2],[99];x=0) == [0]
        @test boundary_residual(dirichlet,[3],[99];x=0) == [1]
        neumann=LinearEqualityBoundary(1,0,3)
        @test boundary_residual(neumann,[99],[3];x=5) == [0]
        robin=LinearEqualityBoundary(2,-3,4)
        @test boundary_residual(robin,[2],[5];x=5) == [0]
        @test boundary_coefficients(robin,5) == (;a=(2,),b=(-3,),c=(4,))
        upper=LinearInequalityBoundary(1,0,2)
        @test boundary_residual(upper,[0,0,0],[1,2,3];x=5) == [-1,0,1]
        lower=LinearInequalityBoundary(-1,0,-2)
        @test boundary_residual(lower,[0,0,0],[3,2,1];x=0) == [-1,0,1]
        @test boundary_residual(lower,[0],[3];x=0) == boundary_residual(lower,[0],[3];x=5)
    end

    @testset "Vector coefficients without a discrete input" begin
        for constructor in (LinearEqualityBoundary,LinearInequalityBoundary)
            rhs=VectorBoundaryRHS(1//3)
            condition=constructor(a=x->[x+1,x+2],b=1//2,c=rhs)
            coefficients=boundary_coefficients(condition,1//2)
            @test coefficients.a == [3//2,5//2]
            @test coefficients.b === (1//2,1//2)
            @test coefficients.c == [11//6,17//6]
            @test boundary_residual(condition,[2,2],[1,1];x=1//2) == [2//3,2//3]
            @test_throws DimensionMismatch boundary_coefficients(condition,1//2;ncomponents=3)
            calls=Ref(0)
            callback=x->(calls[]+=1;[2,3])
            lazy=constructor(1,0,callback)
            @test calls[] == 0
            @test lazy.c === callback
            @test boundary_coefficients(lazy,0).c == [2,3]
            @test calls[] == 1
            @test_throws ArgumentError constructor(0,0,1)
            @test_throws ArgumentError constructor([0,1],[0,0],[1,1])
            @test_throws ArgumentError constructor(Inf,0,1)
            @test_throws ArgumentError constructor(1,NaN,1)
            @test_throws ArgumentError constructor(1,0,1+im)
            @test_throws ArgumentError constructor(1,0,[])
            @test_throws DimensionMismatch constructor([1,1],[0,0,0],1)
            for invalid in (x->NaN,x->[Inf,1],x->1+im,x->zeros(2,2))
                @test_throws ArgumentError boundary_coefficients(constructor(1,0,invalid),0)
            end
            @test_throws ArgumentError boundary_coefficients(constructor(x->[0,1],0,1),0)
            @test_throws ArgumentError boundary_coefficients(constructor(1,0,1),0;ncomponents=0)
            @test_throws DimensionMismatch boundary_residual(constructor(1,0,1),[1,2],[1];x=0)
        end
    end

    @testset "Face collections and geometry integration" begin
        lower=(LinearInequalityBoundary(-1,0,[-2,-3]),LinearEqualityBoundary(0,1,[4,5]))
        upper=(LinearEqualityBoundary(1,0,[6,7]),LinearInequalityBoundary(1,1,[10,10]))
        boundary=RectangularBoundary(lower=collect(lower),upper=upper)
        @test length(boundary) == 2
        @test boundary.lower == lower
        @test boundary.upper == upper
        @test_throws DimensionMismatch RectangularBoundary(lower,upper[1])
        @test_throws ArgumentError RectangularBoundary((),())
        @test_throws ArgumentError RectangularBoundary((:placeholder,),upper[1])
        bomb=(args...)->error("boundary evaluation must not evaluate HJB primitives")
        problem=HJBProblem(bomb,bomb,bomb,bomb;control=bomb,
                           domain=RectangularDomain([0,0],[5,2]),boundary)
        @test boundary_residual(problem,[4,5],[3 4;0 0];x=(0,1),axis=1,side=:lower) == [-1,-1]
        @test boundary_residual(problem,[4,5],[0 0;0 0];x=(1,0),axis=2,side=:lower) == [0,0]
        @test boundary_residual(problem,[4,5],[6 7;0 0];x=(5,1),axis=1,side=:upper) == [0,0]
        @test boundary_residual(problem,[4,5],[0 0;2 3];x=(1,2),axis=2,side=:upper) == [-4,-2]
        for axis in (1,2)
            expected=axis==1 ? [-1,-1] : [0,0]
            @test boundary_residual(problem,[4,5],[3 4;0 0];x=(0,0),axis,side=:lower) == expected
        end
        @test_throws ArgumentError boundary_residual(problem,[4,5],zeros(2,2);x=(1,1),axis=1,side=:lower)
        @test_throws DomainError boundary_residual(problem,[4,5],zeros(2,2);x=(-1,1),axis=1,side=:lower)
        @test_throws ArgumentError boundary_residual(problem,[4,5],zeros(2,2);x=(0,1),axis=3,side=:lower)
        @test_throws ArgumentError boundary_residual(problem,[4,5],zeros(2,2);x=(0,1),axis=1,side=:other)
        @test_throws DimensionMismatch boundary_residual(problem,[4,5],zeros(1,2);x=(0,1),axis=1,side=:lower)
    end
end
