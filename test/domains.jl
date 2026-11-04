@testset "Rectangular domains" begin
    lower=[0.0,0.5]
    domain=RectangularDomain(lower=lower,upper=[5.0,1.5])
    @test domain.lower == (0.0,0.5)
    @test domain.upper == (5.0,1.5)
    @test length(domain) == 2
    @test (1.0,1.0) in domain
    @test [0.0,0.5] in domain
    @test [5.0,1.5] in domain
    @test !([5.1,1.0] in domain)
    @test !(1.0 in domain)
    @test !([1.0] in domain)
    @test !([NaN,1.0] in domain)
    lower[1]=-1.0
    @test domain.lower[1] == 0.0
    interval=RectangularDomain(0,5)
    @test 0 in interval
    @test 5 in interval
    @test (3,) in interval
    @test !(6 in interval)
    exact=RectangularDomain((0//1,),(5//2,))
    @test exact.upper === (5//2,)
    @test_throws DimensionMismatch RectangularDomain([0,0],[1])
    @test_throws ArgumentError RectangularDomain([],[])
    @test_throws ArgumentError RectangularDomain(1,1)
    @test_throws ArgumentError RectangularDomain(2,1)
    @test_throws ArgumentError RectangularDomain(-Inf,1)
    @test_throws ArgumentError RectangularDomain(0,NaN)
    @test_throws ArgumentError RectangularDomain([0],[1+im])
end
