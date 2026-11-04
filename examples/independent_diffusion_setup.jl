using PolicyPDEs

# Two continuous coordinates, two CTMC components, and independent coordinate
# Brownian motions. All five primitives accept continuous state and control;
# the regime label selects the component of the vector value v(x).
# This is a setup example with illustrative coefficients/boundaries, not a solve.
function independent_diffusion_problem()
    return HJBProblem(;
        discount = (x, c, i) -> 0.05 + 0.001*x[1] + 0.01*c,
        flow = (x, c, i) -> log(c) + 0.01*x[2] * i,
        drift = (x, c, i) -> ((i == 1 ? 0.1 : 0.2)*x[2] + 0.03*x[1] - c,
                              0.2*(1.0 - x[2])),
        diffusion = (x, c, i) -> (0.05*x[1], 0.1*x[2]*(1 + 0.01*c)),
        transition = (x, c, i, j) -> (i == 1 ? 0.02 : 0.03)*(1 + 0.1*x[2] + 0.01*c),
        regimes = (1, 2),
        domain = (; assets=(0.0,5.0), productivity=(0.5,1.5)),
        boundary = (;
            assets=(lower=LinearEqualityBoundary(1,0,0), upper=LinearEqualityBoundary(1,0,0)),
            productivity=(lower=LinearEqualityBoundary(1,0,0), upper=LinearEqualityBoundary(1,0,0)),
        ),
    )
end
