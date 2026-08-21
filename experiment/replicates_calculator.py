import math
from statsmodels.stats.power import TTestIndPower

def calculate_doe_replicates(factors, effect_size=0.5, power=0.8):
    # 1. Find total N per "side" of the comparison
    analysis = TTestIndPower()
    n_per_side = analysis.solve_power(effect_size=effect_size, power=power, alpha=0.05)
    
    # 2. Total runs needed = 2 * n_per_side
    total_runs_needed = n_per_side * 2
    
    # 3. Runs in one full design (2^k)
    unique_combinations = 2**factors
    
    # 4. Replicates = Total Runs / Unique Combinations
    replicates = math.ceil(total_runs_needed / unique_combinations)
    return replicates

# For a 5-factor experiment:
print(f"Replicates needed: {calculate_doe_replicates(factors=10)}") 
# This will likely return 4, not 64!