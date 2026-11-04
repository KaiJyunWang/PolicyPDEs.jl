using PolicyPDEs
using Test

test_boundary(d) = RectangularBoundary(
    lower=ntuple(_ -> LinearEqualityBoundary(1,0,0),d),
    upper=ntuple(_ -> LinearEqualityBoundary(1,0,0),d),
)

@testset "PolicyPDEs.jl" begin
    include("domains.jl")
    include("boundaries.jl")
    include("problems.jl")
    include("evaluation.jl")
end
