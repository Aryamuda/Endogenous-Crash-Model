import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Ensure src is discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.model import MarketCrashModel
from src.diagnostics import compute_diagnostics
from src.visualization import CB_COLORS

def run_alpha_study():
    """
    Sensitivity study for the projection parameter alpha.
    Tests if relative precursors are invariant to scaling.
    """
    print("="*60)
    print("ALPHA-SENSITIVITY ROBUSTNESS STUDY")
    print("="*60)

    # Configuration
    alphas = [10, 20, 30]
    K_safe = 0.5
    K_crit = 1.9
    N = 1000
    sigma = 0.5
    T = 100
    dt = 0.05
    lambda_0 = 0.5
    seed = 42

    results = []
    
    print(f"{'Alpha':<10} | {'Base AC1':<10} | {'Crit AC1':<10} | {'AC1 Inc %':<10} | {'Var Inf %':<10}")
    print("-" * 65)

    for alpha in alphas:
        # 1. Simulate Safe Regime
        model_s = MarketCrashModel(N=N, sigma=sigma, alpha=alpha, lambda_base=lambda_0, seed=seed)
        r_s = model_s.evolve(K_safe, T, dt)
        lam_s = np.array([model_s.compute_jump_intensity(r) for r in r_s])
        diag_s = compute_diagnostics(lam_s)
        
        # 2. Simulate Critical Regime
        model_c = MarketCrashModel(N=N, sigma=sigma, alpha=alpha, lambda_base=lambda_0, seed=seed)
        r_c = model_c.evolve(K_crit, T, dt)
        lam_c = np.array([model_c.compute_jump_intensity(r) for r in r_c])
        diag_c = compute_diagnostics(lam_c)
        
        # 3. Calculate Relative Metrics
        ac_inc = ((diag_c['Autocorrelation'] - diag_s['Autocorrelation']) / diag_s['Autocorrelation']) * 100
        var_inf = ((diag_c['Variance'] - diag_s['Variance']) / diag_s['Variance']) * 100
        
        results.append({
            'alpha': alpha,
            'ac_safe': diag_s['Autocorrelation'],
            'ac_crit': diag_c['Autocorrelation'],
            'ac_inc': ac_inc,
            'var_inf': var_inf
        })
        
        print(f"{alpha:<10} | {diag_s['Autocorrelation']:<10.3f} | {diag_c['Autocorrelation']:<10.3f} | {ac_inc:<10.1f} | {var_inf:<10.1f}")

    # Generate Visualization (Ramp Scenario)
    print("\nGenerating robustness figure (Figure_alpha_robustness.pdf)...")
    steps = int(T / dt)
    K_ramp = np.linspace(0, 4, steps)
    
    plt.figure(figsize=(10, 6))
    colors = [CB_COLORS['blue'], CB_COLORS['orange'], CB_COLORS['green']]
    
    for i, alpha in enumerate(alphas):
        model_r = MarketCrashModel(N=N, sigma=sigma, alpha=alpha, lambda_base=lambda_0, seed=seed)
        r_hist = []
        for K in K_ramp:
            r_hist.append(model_r.step(K, dt))
        lam_hist = np.array([model_r.compute_jump_intensity(r) for r in r_hist])
        v_hist = pd.Series(lam_hist).rolling(window=200).var()
        plt.plot(v_hist, label=f'alpha = {alpha}', color=colors[i], linewidth=2)

    plt.axvline(x=steps/2, color='red', linestyle='--', alpha=0.5, label='Transition Zone (K~2)')
    plt.title('Alpha Robustness: Relative Signal Invariance', fontsize=13, fontweight='bold')
    plt.xlabel('Time Steps (Increasing Coupling K)', fontsize=11)
    plt.ylabel('Rolling Variance of lambda(t)', fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Figure_alpha_robustness.pdf', format='pdf', bbox_inches='tight')
    print("✓ Figure saved: Figure_alpha_robustness.pdf")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_alpha_study()
