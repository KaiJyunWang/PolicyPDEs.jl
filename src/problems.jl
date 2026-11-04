"""
    HJBProblem(discount, drift, diffusion, flow; domain, boundary,
               regimes=(1,), transition=nothing)
    HJBProblem(; discount, drift, diffusion, flow, domain, boundary,
               regimes=(1,), transition=nothing)

Describe a stationary vector-valued HJB `v: R^d -> R^m` with independent
coordinate Brownian motions and CTMC switching. Discrete states index value
components, not continuous domain axes.

All five user-defined primitives are callable structs or functions of
`(x, control)`, with no discrete-state argument. Their output shapes are:

- `discount(x, control)`: length-`m` vector of discount rates.
- `flow(x, control)`: length-`m` vector of instantaneous payoffs.
- `drift(x, control)`: `d × m` matrix; entry `[k,i]` is coordinate `k`'s
  drift in regime `i`.
- `diffusion(x, control)`: `d × m` matrix of coordinate volatilities.
  Entry `[k,i]` contributes `sigma[k,i]^2 / 2` to the pure second derivative.
  Columns index regimes, not Brownian shocks. Coordinate shocks are independent;
  mixed derivatives and correlated innovations are outside the initial scope.
- `transition(x, control)`: `m × m` matrix of off-diagonal intensities;
  `[i,j]` is the nonnegative rate from regime `i` to `j`. Diagonal entries
  must be zero. This is a rate matrix, not the full CTMC generator; the generator
  diagonal is minus the outgoing rate sum. With `transition=nothing`, all
  switching rates are zero.

Vectors must retain length `m` and matrices their stated shapes even when
`d=1` or `m=1`; tuples are also accepted for the vector outputs. Continuous
coordinates follow domain-axis order; value entries, coefficient columns,
and transition rows/columns follow `regimes` order.

For a supplied control, component `i` has residual

    flow[i] - discount[i] * v[i]
      + sum(drift[k,i] * partial_k(v_i)
            + diffusion[k,i]^2 * partial_kk(v_i) / 2 for k in 1:d)
      + sum(transition[i,j] * (v[j] - v[i]) for j != i).

The control is passed unchanged to every callback. A vector can contain one
control per regime, or a matrix can hold several controls per regime. For
row-wise policy improvement, output component/column `i` and transition row
`i` should depend on the source regime's control only. A shared candidate
control is also possible if the callbacks implement it. Control feasibility
and optimization belong to the future solver. Control-dependent discounting
stays inside the maximization of the complete residual.

Use a scalar continuous state in one dimension or a tuple/vector in multiple
dimensions. `domain` and `boundary` are required; linear boundary conditions
use axis-keyed named tuples with `lower` and `upper` entries. Their coefficient
callback interface remains `(x, regime)`; it is separate from HJB primitives.
`regimes` is a nonempty tuple/vector of unique labels, defaulting to `(1,)`.

Construction stores inputs without copying or evaluating callbacks and checks
regime labels only. Mutable inputs remain shared; do not mutate regime order.
[`hjb_coefficients`](@ref) validates evaluated shapes, finite values, and rates.
[`hjb_residual`](@ref) evaluates all value components from supplied derivatives.
No spatial discretizer or PDE solver is implemented yet.

# Example

```julia
problem = HJBProblem(;
    discount = (x, c) -> [0.05, 0.05],
    flow = (x, c) -> log.(c),
    drift = (x, c) -> reshape([0.1, 0.2] .+ 0.03*x .- c, 1, 2),
    diffusion = (x, c) -> zeros(1, 2),
    transition = (x, c) -> [0.0 0.02; 0.03 0.0],
    regimes = (:low, :high),
    domain = (; assets = (0.0, 5.0)),
    boundary = (; assets = (
        lower = LinearEqualityBoundary(1, 0, 0),
        upper = LinearEqualityBoundary(1, 0, 0),
    )),
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
