"""
    HJBProblem(discount, drift, diffusion, flow; domain, boundary,
               regimes=(1,), transition=nothing)
    HJBProblem(; discount, drift, diffusion, flow, domain, boundary,
               regimes=(1,), transition=nothing)

Describe one stationary HJB equation, or a finite system of HJB equations
coupled by Markov switching, before spatial discretization. For regime `i`,

    0 = max_a { flow(x, a, i) - discount(x, a, i) * v_i(x)
                + drift(x, a, i) ⋅ gradient(v_i)(x)
                + (1/2) * tr(sigma * sigma' * hessian(v_i)(x))
                + sum(q_ij(x, a) * (v_j(x) - v_i(x)) for j != i) },

where `sigma = diffusion(x, a, i)` and
`q_ij(x, a) = transition(x, a, i, j)`. Policy feasibility and optimization are
part of the future discretization/solver interfaces.

# Components

Supply each component as a callable user-defined struct or a function. No
package supertype is required. All four components use `(x, a, i)`, where
`x` is the continuous state, `a` is the control in the source regime, and `i`
is a label from `regimes`. State and control representations are left to the
discretizer.

- `discount(x, a, i)`: scalar discount rate. A constant rate is represented by
  a callable that ignores its arguments.
- `drift(x, a, i)`: continuous-state drift; scalar in one dimension, vector in
  multiple dimensions.
- `diffusion(x, a, i)`: volatility, **not** variance or the coefficient already
  multiplied by `1/2`. Return a scalar in one dimension or a state-by-shock
  matrix in multiple dimensions. Return zero volatility for deterministic
  dynamics.
- `flow(x, a, i)`: scalar instantaneous payoff to be maximized.

# Problem data and coupling

- `domain`: continuous state-space description, independent of a numerical grid.
- `boundary`: explicit boundary data; there is no implicit boundary condition.
  For linear conditions, use axis-keyed named tuples with `lower` and `upper`
  entries holding [`LinearEqualityBoundary`](@ref) or
  [`LinearInequalityBoundary`](@ref). Coefficients may depend on state and regime.
- `regimes`: a nonempty tuple or vector of unique regime labels. Defaults to
  `(1,)` for a single PDE. The order determines the ordering of equations.
- `transition`: callable `transition(x, a, i, j)` returning the finite,
  nonnegative intensity **from `i` to `j`**, for `i != j`. It may depend on
  state and source-regime control. Supply off-diagonal rates only: the generator
  diagonal is minus the sum of outgoing rates. With `nothing`, all switching
  rates are zero, so multiple regimes describe uncoupled equations. A single
  regime has no switching term and never requires a diagonal rate evaluation.
  Switching preserves the continuous state `x`.

The problem stores inputs with their concrete types, without copying or
evaluating components. Mutable inputs remain shared; do not mutate regime
membership or ordering after construction. Construction checks only the regime
collection. Callback return shapes, rate signs, numerical boundary enforcement, and
mathematical well-posedness must be checked by future numerical stages at
evaluation points. Grids, initial guesses, pseudo-time steps, and tolerances
belong to later stages. No discretizer or solver is implemented yet.

# Example

```julia
discount = (x, c, i) -> 0.05
drift = (x, c, i) -> (i == 1 ? 0.1 : 0.2) + 0.03 * x - c
diffusion = (x, c, i) -> 0.0
flow = (x, c, i) -> log(c)
transition = (x, c, i, j) -> i == 1 ? 0.02 : 0.03

problem = HJBProblem(discount, drift, diffusion, flow;
    domain = (; assets = (0.0, 5.0)),
    boundary = (; assets = (
        lower = LinearInequalityBoundary(a=-1, b=0,
            c=(x, i) -> -inv((i == 1 ? 0.1 : 0.2) + 0.03*x)),
        upper = LinearEqualityBoundary(a=1, b=0,
            c=(x, i) -> inv((i == 1 ? 0.1 : 0.2) + 0.03*x)),
    )),
    regimes = (1, 2), transition,
)
```
"""
struct HJBProblem{R,M,S,F,D,B,I,Q}
    discount::R
    drift::M
    diffusion::S
    flow::F
    domain::D
    boundary::B
    regimes::I
    transition::Q

    function HJBProblem(discount, drift, diffusion, flow, domain, boundary,
                        regimes, transition)
        regimes isa Union{Tuple,AbstractVector} ||
            throw(ArgumentError("regimes must be a nonempty tuple or vector of unique labels"))
        isempty(regimes) && throw(ArgumentError("regimes must not be empty"))
        allunique(regimes) || throw(ArgumentError("regime labels must be unique"))
        return new{typeof(discount),typeof(drift),typeof(diffusion),typeof(flow),
                   typeof(domain),typeof(boundary),typeof(regimes),typeof(transition)}(
            discount, drift, diffusion, flow, domain, boundary, regimes, transition,
        )
    end
end

function HJBProblem(discount, drift, diffusion, flow; domain, boundary,
                    regimes = (1,), transition = nothing)
    return HJBProblem(discount, drift, diffusion, flow, domain, boundary,
                      regimes, transition)
end

function HJBProblem(; discount, drift, diffusion, flow, domain, boundary,
                    regimes = (1,), transition = nothing)
    return HJBProblem(discount, drift, diffusion, flow; domain, boundary,
                      regimes, transition)
end
