"""
    HJBProblem(; discount, drift, diffusion, flow, domain, boundary, transition = nothing)
    HJBProblem(discount, drift, diffusion, flow; domain, boundary, transition = nothing)

Stationary vector-valued HJB on a rectangle with independent coordinate Brownian
motions and CTMC switching. Every coefficient callback has the signature
`coefficient(x, v, Dv, D2v)`. There is no policy or control argument.

For `d` coordinates and `m` value components, `v` is a length-`m` vector/tuple
and `Dv`/`D2v` are `d`-by-`m` matrices of first/pure-second derivatives.
Callbacks return:

  - `discount`, `flow`: length-`m` vectors/tuples;
  - `drift`, `diffusion`: `d`-by-`m` matrices (coordinate by value component);
  - `transition`: `m`-by-`m` off-diagonal rates, source by destination, zero diagonal.

The diffusion term is `sigma[k, i]^2 * D2v[k, i] / 2`. Mixed derivatives are
unsupported. With `transition = nothing`, switching is absent.

Inputs are read-only. Coefficients must be finite and retain their shapes.
Callbacks may perform model-specific calculations internally, but the package
neither accepts nor returns a control vector. The user is responsible for
encoding economically appropriate coefficients, including any maximization.

For component `i`, the equation is:

```julia
discount[i] * v[i] = flow[i] +
    sum(drift[k, i] * Dv[k, i] + diffusion[k, i]^2 * D2v[k, i] / 2 for k = 1:d) +
    sum(transition[i, j] * (v[j] - v[i]) for j = 1:m if j != i)
```

Omit the last term when `transition = nothing`.
The solver freezes the evaluated coefficients for each sparse linear solve. The
finite-difference solver evaluates forward/backward derivatives and applies
its drift-sign selection rules before freezing coefficients.

`x` is scalar in one dimension and a tuple/vector otherwise. `domain` must be a
`RectangularDomain` and `boundary` a matching `RectangularBoundary`. Construction
checks geometry without evaluating callbacks. LinearBoundary and inequality
coefficient callbacks use `boundary_coefficient(x)`. NonlinearBoundary callbacks
use `boundary_coefficient(x, v, Dv, D2v)`, like the PDE coefficients.
"""
struct HJBProblem{R, M, S, F, D, B, Q}
    discount::R
    drift::M
    diffusion::S
    flow::F
    domain::D
    boundary::B
    transition::Q

    function HJBProblem(discount, drift, diffusion, flow, domain, boundary, transition)
        domain isa RectangularDomain ||
            throw(ArgumentError("domain must be a RectangularDomain"))
        boundary isa RectangularBoundary ||
            throw(ArgumentError("boundary must be a RectangularBoundary"))
        length(domain) == length(boundary) || throw(
            DimensionMismatch(
                "domain dimension must match lower/upper boundary face counts",
            ),
        )
        new{
            typeof(discount),
            typeof(drift),
            typeof(diffusion),
            typeof(flow),
            typeof(domain),
            typeof(boundary),
            typeof(transition),
        }(
            discount,
            drift,
            diffusion,
            flow,
            domain,
            boundary,
            transition,
        )
    end
end

function HJBProblem(
    discount,
    drift,
    diffusion,
    flow;
    domain,
    boundary,
    transition = nothing,
)
    HJBProblem(discount, drift, diffusion, flow, domain, boundary, transition)
end

function HJBProblem(;
    discount,
    drift,
    diffusion,
    flow,
    domain,
    boundary,
    transition = nothing,
)
    HJBProblem(discount, drift, diffusion, flow; domain, boundary, transition)
end
