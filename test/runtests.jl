using PolicyPDEs
using Test

test_boundary(d) = RectangularBoundary(
    lower=ntuple(_ -> LinearEqualityBoundary(1,0,0),d),
    upper=ntuple(_ -> LinearEqualityBoundary(1,0,0),d),
)

test_control(v,Dv,D2v,x) = ones(length(v))

@testset "PolicyPDEs.jl" begin
    include("domains.jl")
    include("boundaries.jl")
    include("problems.jl")
    include("evaluation.jl")
    include("controls.jl")
end
