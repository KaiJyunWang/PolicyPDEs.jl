struct RegimeBoundaryRHS{T}
    base::T
end
(rhs::RegimeBoundaryRHS)(x, regime) = rhs.base + x + regime

@testset "Linear boundaries" begin
    @testset "Equality, Dirichlet, Neumann, and Robin forms" begin
        dirichlet = @inferred LinearEqualityBoundary(a = 0, b = 1, c = 2)
        @test boundary_residual(dirichlet, 2, 99; x = 0) == 0
        @test boundary_residual(dirichlet, 3, 99; x = 0) == 1
        neumann = @inferred LinearEqualityBoundary(1, 0, 3)
        @test boundary_residual(neumann, 99, 3; x = 5) == 0
        robin = LinearEqualityBoundary(a = 2, b = -3, c = 4)
        @test boundary_residual(robin, 2, 5; x = 5) == 0
        @test boundary_residual(robin, 2, 4; x = 5) == -2
        @test boundary_coefficients(robin, 5) == (; a = 2, b = -3, c = 4)
    end

    @testset "Inequality orientation and endpoint derivative convention" begin
        upper_bound = @inferred LinearInequalityBoundary(a = 1, b = 0, c = 2)
        @test boundary_residual(upper_bound, 0, 1; x = 5) < 0
        @test boundary_residual(upper_bound, 0, 2; x = 5) == 0
        @test boundary_residual(upper_bound, 0, 3; x = 5) > 0
        lower_bound = @inferred LinearInequalityBoundary(-1, 0, -2)
        @test boundary_residual(lower_bound, 0, 3; x = 0) < 0
        @test boundary_residual(lower_bound, 0, 2; x = 0) == 0
        @test boundary_residual(lower_bound, 0, 1; x = 0) > 0
        # The derivative is with respect to increasing x at either endpoint.
        @test boundary_residual(lower_bound, 0, 3; x = 0) ==
              boundary_residual(lower_bound, 0, 3; x = 5)
        mixed = LinearInequalityBoundary(a = 2, b = 3, c = 10)
        @test boundary_residual(mixed, 2, 1; x = 0) == -2
    end

    @testset "State/regime-dependent coefficients and callable structs" begin
        for constructor in (LinearEqualityBoundary, LinearInequalityBoundary)
            boundary = constructor(a = (x, i) -> x + i, b = 1 // 2,
                                   c = RegimeBoundaryRHS(1 // 3))
            coefficients = @inferred boundary_coefficients(boundary, 1 // 2, 2)
            @test coefficients == (; a = 5 // 2, b = 1 // 2, c = 17 // 6)
            @test coefficients.c isa Rational{Int}
            @test boundary_residual(boundary, 2, 1; x = 1 // 2, regime = 2) == 2 // 3
        end
    end

    @testset "Validation without evaluating user callbacks during setup" begin
        for constructor in (LinearEqualityBoundary, LinearInequalityBoundary)
            calls = Ref(0)
            callback = (x, i) -> (calls[] += 1; 2)
            boundary = constructor(a = 1, b = 0, c = callback)
            @test calls[] == 0
            @test boundary.c === callback
            @test boundary_coefficients(boundary, 0).c == 2
            @test calls[] == 1
            @test_throws ArgumentError constructor(0, 0, 1)
            @test_throws ArgumentError constructor(Inf, 0, 1)
            @test_throws ArgumentError constructor(1, NaN, 1)
            @test_throws ArgumentError constructor(1, 0, Inf)
            @test_throws ArgumentError constructor(1, 0, 1 + im)
            for invalid in ((x, i) -> NaN, (x, i) -> [1, 2], (x, i) -> 1 + im)
                invalid_boundary = constructor(1, 0, invalid)
                @test_throws ArgumentError boundary_coefficients(invalid_boundary, 0)
            end
            degenerate = constructor((x, i) -> 0, (x, i) -> 0, 1)
            @test_throws ArgumentError boundary_coefficients(degenerate, 0)
        end
    end
end
