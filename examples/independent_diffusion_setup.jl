using PolicyPDEs

# Two coordinates and two value components. The user-defined policy update
# maximizes the HJB residual on a bounded positive consumption grid.
# This illustrates a supplied numerical maximizer, not a built-in optimizer.
function independent_diffusion_problem()
    discount=(x,c)->0.05 .+ 0.001*x[1] .+ 0.01 .* c
    flow=(x,c)->log.(c) .+ 0.01*x[2] .* [1,2]
    drift=(x,c)->[0.1*x[2]+0.03*x[1]-c[1] 0.2*x[2]+0.03*x[1]-c[2];
                  0.2*(1-x[2]) 0.2*(1-x[2])]
    diffusion=(x,c)->[0.05*x[1] 0.05*x[1];
                      0.1*x[2]*(1+0.01*c[1]) 0.1*x[2]*(1+0.01*c[2])]
    transition=(x,c)->[0.0 0.02*(1+0.1*x[2]+0.01*c[1]);
                       0.03*(1+0.1*x[2]+0.01*c[2]) 0.0]
    control=(v,Dv,D2v,x)->begin
        best=fill(-Inf,length(v))
        chosen=zeros(length(v))
        for candidate in 0.1:0.1:1.0
            trial=fill(candidate,length(v))
            rho,u=discount(x,trial),flow(x,trial)
            mu,sigma=drift(x,trial),diffusion(x,trial)
            rates=transition(x,trial)
            for i in eachindex(v)
                score=u[i]-rho[i]*v[i]
                score+=sum(mu[k,i]*Dv[k,i]+sigma[k,i]^2*D2v[k,i]/2 for k in axes(Dv,1))
                score+=sum(rates[i,j]*(v[j]-v[i]) for j in eachindex(v))
                if score > best[i]
                    best[i],chosen[i]=score,candidate
                end
            end
        end
        chosen
    end
    return HJBProblem(;discount,flow,drift,diffusion,transition,control,
        domain=RectangularDomain(lower=[0.0,0.5],upper=[5.0,1.5]),
        boundary=RectangularBoundary(
            lower=(LinearEqualityBoundary(1,0,0),LinearEqualityBoundary(1,0,0)),
            upper=(LinearEqualityBoundary(1,0,0),LinearEqualityBoundary(1,0,0)),
        ),
    )
end
