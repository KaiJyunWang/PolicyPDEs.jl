module PolicyPDEs

export HJBProblem
export RectangularDomain, RectangularBoundary
export hjb_coefficients, hjb_residual
export update_control
export LinearEqualityBoundary, LinearInequalityBoundary
export boundary_coefficients, boundary_residual

include("domains.jl")
include("boundaries.jl")
include("problems.jl")
include("evaluation.jl")

end
