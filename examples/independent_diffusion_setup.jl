using PolicyPDEs

# Two continuous coordinates (rows), two CTMC components (columns).
# c is a two-entry vector; each coefficient column and outgoing transition
# row uses its source component's control. No callback takes a regime label.
function independent_diffusion_problem()
    return HJBProblem(;
        discount=(x,c)->0.05 .+ 0.001*x[1] .+ 0.01 .* c,
        flow=(x,c)->log.(c) .+ 0.01*x[2] .* [1,2],
        drift=(x,c)->[0.1*x[2]+0.03*x[1]-c[1] 0.2*x[2]+0.03*x[1]-c[2];
                      0.2*(1-x[2]) 0.2*(1-x[2])],
        diffusion=(x,c)->[0.05*x[1] 0.05*x[1];
                          0.1*x[2]*(1+0.01*c[1]) 0.1*x[2]*(1+0.01*c[2])],
        transition=(x,c)->[0.0 0.02*(1+0.1*x[2]+0.01*c[1]);
                           0.03*(1+0.1*x[2]+0.01*c[2]) 0.0],
        regimes=(1,2),
        domain=(; assets=(0.0,5.0), productivity=(0.5,1.5)),
        boundary=(;
            assets=(lower=LinearEqualityBoundary(1,0,0), upper=LinearEqualityBoundary(1,0,0)),
            productivity=(lower=LinearEqualityBoundary(1,0,0), upper=LinearEqualityBoundary(1,0,0)),
        ),
    )
end
