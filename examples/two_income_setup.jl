using PolicyPDEs

# Setup for two income regimes with Poisson switching, following the HJB
# structure in Achdou et al. (2022), Review of Economic Studies 89(1), 45-86.
# The illustrative parameters below are not a replication calibration.
# https://benjaminmoll.com/wp-content/uploads/2019/07/HACT.pdf

struct ConstantDiscount{T}
    rate::T
end
(discount::ConstantDiscount)(wealth, consumption, regime) = discount.rate

struct AssetDrift{T,Y}
    interest::T
    income::Y
end
(drift::AssetDrift)(wealth, consumption, regime) =
    drift.income[regime] + drift.interest * wealth - consumption

struct ZeroVolatility end
(::ZeroVolatility)(wealth, consumption, regime) = 0.0

struct LogFlow end
(::LogFlow)(wealth, consumption, regime) = log(consumption)

struct IncomeTransitions{T}
    low_to_high::T
    high_to_low::T
end
function (rates::IncomeTransitions)(wealth, consumption, from, to)
    from == 1 && to == 2 && return rates.low_to_high
    from == 2 && to == 1 && return rates.high_to_low
    throw(ArgumentError("expected an off-diagonal transition between regimes 1 and 2"))
end

function two_income_problem()
    drift = AssetDrift(0.03, (0.1, 0.2))
    marginal = (wealth, regime) -> inv(drift.income[regime] + drift.interest * wealth)
    boundary = (; assets = (
        # Log utility and no borrowing: v_x >= u'(income + r*x) at x = 0.
        lower = LinearInequalityBoundary(a = -1, b = 0,
                                         c = (wealth, regime) -> -marginal(wealth, regime)),
        # Illustrative upper truncation: impose the zero-saving marginal value.
        upper = LinearEqualityBoundary(a = 1, b = 0, c = marginal),
    ))
    return HJBProblem(
        ConstantDiscount(0.05),
        drift,
        ZeroVolatility(),
        LogFlow();
        domain = (; assets = (0.0, 5.0)),
        boundary,
        regimes = (1, 2),
        transition = IncomeTransitions(0.02, 0.03),
    )
end
