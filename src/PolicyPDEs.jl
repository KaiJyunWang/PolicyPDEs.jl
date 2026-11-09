module PolicyPDEs

using LinearAlgebra
using SparseArrays
import LinearSolve
import LinearSolve: solve, discretize

export HJBProblem
export RectangularDomain, RectangularBoundary
export hjb_coefficients, hjb_residual
export AbstractSpatialDiscretization, FiniteDifference, TensorGrid
export equidistant, fd_stencil, discretize, DiscretizedHJBProblem
export LinearBoundary, LinearInequalityBoundary
export NonlinearBoundary
export StateConstraintBoundary
export boundary_coefficients, boundary_residual
export spatial_operators, assemble_operator, HJBSolution, solve

include("domains.jl")
include("boundaries.jl")
include("problems.jl")
include("evaluation.jl")
include("discretization.jl")
include("operators.jl")
include("solvers.jl")

end
