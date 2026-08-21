from statsmodels.stats.power import FTestAnovaPower

def estimate_regression_n(effect_size, k_predictors, alpha=0.05, power=0.80):
    analysis = FTestAnovaPower()
    
    # solve_power for F-test (Regression)
    # df_num = number of predictors (k)
    # df_denom = N - k - 1 (what we are solving for)
    n_total = analysis.solve_power(
        effect_size=effect_size, 
        k_groups=k_predictors + 1,  # Number of predictors + intercept
        alpha=alpha, 
        power=power
    )
    return n_total

# Parameters
f2_small = 0.2   # Small effect

n_required = estimate_regression_n(f2_small, k_predictors=32)
print(f"Required total sample size: {round(n_required)}")