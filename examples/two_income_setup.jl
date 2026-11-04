using PolicyPDEs

# Two-income HJB structure of Achdou et al. (2022), REStud 89(1), 45-86.
# Illustrative parameters and upper truncation, not a replication calibration.
# https://benjaminmoll.com/wp-content/uploads/2019/07/HACT.pdf
# HJB primitives take (wealth, consumption), with consumption a two-entry vector.

struct ConstantDiscount{T}
    rates::T
end
(discount::ConstantDiscount)(wealth, consumption) = discount.rates

struct AssetDrift{T,Y}
    interest::T
    income::Y
end
(drift::AssetDrift)(wealth, consumption) =
    reshape(drift.income .+ drift.interest * wealth .- consumption, 1, 2)

struct ZeroVolatility end
(::ZeroVolatility)(wealth, consumption) = zeros(1, 2)

struct LogFlow end
(::LogFlow)(wealth, consumption) = log.(consumption)

struct IncomeTransitions{T}
    low_to_high::T
    high_to_low::T
end
(rates::IncomeTransitions)(wealth, consumption) =
    [0 rates.low_to_high; rates.high_to_low 0]

function two_income_problem()
    drift = AssetDrift(0.03, [0.1, 0.2])
    marginal = wealth -> inv.(drift.income .+ drift.interest * wealth)
    boundary = RectangularBoundary(
        lower=LinearInequalityBoundary(a=-1,b=0,c=x->-marginal(x)),
        upper=LinearEqualityBoundary(a=1,b=0,c=marginal),
    )
    return HJBProblem(
        ConstantDiscount([0.05, 0.05]), drift, ZeroVolatility(), LogFlow();
        domain=RectangularDomain(lower=[0.0],upper=[5.0]), boundary,
        transition=IncomeTransitions(0.02,0.03),
    )
end
