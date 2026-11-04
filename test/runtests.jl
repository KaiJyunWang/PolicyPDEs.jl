using PolicyPDEs
using Test

@testset "PolicyPDEs.jl" begin
    include("boundaries.jl")
    include("problems.jl")
    include("evaluation.jl")
end
