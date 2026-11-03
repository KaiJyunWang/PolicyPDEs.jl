module PolicyPDEs

export HJBProblem
export LinearEqualityBoundary, LinearInequalityBoundary
export boundary_coefficients, boundary_residual

include("boundaries.jl")
include("problems.jl")

end
