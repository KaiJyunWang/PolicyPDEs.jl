using PolicyPDEs
using Test
using LinearAlgebra
using SparseArrays

test_boundary(d) = RectangularBoundary(
    lower = ntuple(_ -> LinearBoundary(1, 0, 0), d),
    upper = ntuple(_ -> LinearBoundary(1, 0, 0), d),
)


@testset "PolicyPDEs.jl" begin
    include("domains.jl")
    include("boundaries.jl")
    include("problems.jl")
    include("evaluation.jl")
    include("controls.jl")
    include("discretization.jl")
    include("solvers.jl")
    include("nonlinear_boundaries.jl")
end
