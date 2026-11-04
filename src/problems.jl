"""
    HJBProblem(discount, drift, diffusion, flow; domain, boundary,
               regimes=(1,), transition=nothing)
    HJBProblem(; discount, drift, diffusion, flow, domain, boundary,
               regimes=(1,), transition=nothing)

Describe a stationary vector-valued HJB `v: R^d -> R^m`, where `d` is the
continuous state dimension and `m = length(regimes)` is the number of CTMC
states. The discrete state indexes a component of `v(x)`; it is not a
continuous coordinate or another grid axis. For component `i`,

    0 = max_a { flow(x, a, i) - discount(x, a, i) * v_i(x)
                + sum(mu_k * partial_k(v_i) for k in 1:d)
                + (1/2) * sum(sigma_k^2 * partial_kk(v_i) for k in 1:d)
                + sum(q_ij(x, a) * (v_j(x) - v_i(x)) for j != i) },

where `mu = drift(x, a, i)`, `sigma = diffusion(x, a, i)`, and
`q_ij(x, a) = transition(x, a, i, j)`. Policy feasibility and optimization are
part of the future discretization/solver interfaces.

# Components

Supply each component as a callable user-defined struct or a function. No
package supertype is required. All four components use `(x, a, i)`, where
`x` is the continuous state, `a` is the control in the source regime, and `i`
is a label from `regimes`. Use a scalar state in one dimension or a tuple/vector
of continuous coordinates; the control may itself be a structured object.

- `discount(x, a, i)`: scalar discount rate. A constant rate is represented by
  a callable that ignores its arguments.
- `drift(x, a, i)`: one drift per continuous coordinate, as a tuple/vector.
  A scalar is also accepted in one dimension.
- `diffusion(x, a, i)`: volatility, **not** variance or the coefficient already
  multiplied by `1/2`. Return one volatility per continuous coordinate as a
  tuple/vector; a scalar is also accepted in one dimension. The initial scope
  is diagonal state diffusion: each coordinate is driven by its own independent
  Brownian motion, with no mixed second derivatives. Matrix-valued diffusion
  is not supported. Return zero in deterministic coordinates.
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
collection. [`hjb_coefficients`](@ref) checks coefficient shapes and finite
values at supplied states and controls. [`hjb_residual`](@ref) evaluates the
fixed-control equation from user-supplied values and derivatives, including
nonnegative CTMC rates. Numerical boundary enforcement and mathematical
well-posedness remain numerical-stage responsibilities. Grids, initial guesses,
pseudo-time steps, and tolerances belong to later stages. No discretizer or
solver is implemented yet.

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
        regimes isa AbstractVector && Base.require_one_based_indexing(regimes)
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
