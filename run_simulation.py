import numpy as np
from src.model import (
    KuramotoModel, StochasticKuramotoModel, MarketCrashModel,
    phase_transition_sweep, compare_regimes
)
from src.diagnostics import compute_diagnostics, compare_scenarios
from src.visualization import (
    plot_block1, plot_stochastic_precursors, plot_market_projection,
    plot_counterfactuals, plot_robustness
)


def run_block1(save_figs=True):
    """
    Block 1: Phase Transition and Time Evolution.
    """
    print("=" * 60)
    print("BLOCK 1: Phase Transition Demonstration")
    print("=" * 60)
    
    # Configuration
    N = 1000
    dt = 0.05
    T = 50
    
    # 1. Phase Transition Sweep
    K_values = np.linspace(0, 5, 20)
    print("Running phase transition sweep...")
    r_steady = phase_transition_sweep(K_values, N=N, T=T, dt=dt)
    
    # 2. Time Evolution for Specific Regimes
    K_regimes = [0.5, 2.0, 4.0]
    labels = ['Sub-critical ($K=0.5$)', 'Critical ($K=2.0$)', 'Super-critical ($K=4.0$)']
    colors = None  # Color-blind safe palette
    
    print("Running time evolution comparison...")
    time_axis, time_series = compare_regimes(K_regimes, N=N, T=T, dt=dt)
    
    # Visualization
    save_path = "Figure1.pdf" if save_figs else None
    plot_block1(K_values, r_steady, time_axis, time_series, labels, colors, save_path)
    
    print("✓ Block 1 completed\n")


def run_block2(save_figs=True):
    """
    Block 2: Stochastic Precursor Detection.
    """
    print("=" * 60)
    print("BLOCK 2: Stochastic Precursor Detection")
    print("=" * 60)
    
    # Configuration
    N = 1000
    T = 100
    dt = 0.05
    sigma = 0.5
    steps = int(T / dt)
    
    scenarios = {
        'Safe': 0.5,
        'Critical': 1.9
    }
    
    # Initialization
    from scipy.stats import cauchy
    np.random.seed(42)
    omega = cauchy.rvs(loc=0, scale=1, size=N)
    theta_init = np.random.uniform(0, 2*np.pi, N)
    
    def compute_order_parameter(theta):
        z = np.mean(np.exp(1j * theta))
        return np.abs(z), np.angle(z)
    
    def stochastic_kuramoto_step(theta, omega, K, N, dt, sigma):
        """Euler-Maruyama Step"""
        r, psi = compute_order_parameter(theta)
        
        # Deterministic drift
        drift = omega + K * r * np.sin(psi - theta)
        
        # Stochastic diffusion (Wiener process)
        diffusion = sigma * np.random.normal(0, np.sqrt(dt), N)
        
        dtheta = drift * dt + diffusion
        return theta + dtheta, r
    
    # Run simulations
    results = {}
    for name, K in scenarios.items():
        print(f"Simulating {name} Regime (K={K})...")
        theta = theta_init.copy()
        r_series = []
        
        for _ in range(steps):
            theta, r = stochastic_kuramoto_step(theta, omega, K, N, dt, sigma)
            r_series.append(r)
        
        results[name] = np.array(r_series)
    
    # Diagnostics
    print("\n--- STOCHASTIC PRECURSOR RESULTS ---")
    diagnostics = compare_scenarios(results, window=200, burn_in_fraction=0.2)
    print(diagnostics)
    
    # Visualization
    save_path = "Figure2.pdf" if save_figs else None
    plot_stochastic_precursors(results, scenarios, steps, save_path)
    
    print("✓ Block 2 completed\n")


def run_block3(save_figs=True):
    """
    Block 3: Market Projection Mechanism.
    """
    print("=" * 60)
    print("BLOCK 3: Market Projection Mechanism")
    print("=" * 60)
    
    # Configuration
    N = 1000
    T = 200
    dt = 0.05
    sigma = 0.5
    alpha = 20.0
    lambda_base = 0.5
    steps = int(T / dt)
    
    # Dynamic Coupling Ramp
    K_ramp = np.linspace(0, 4, steps)
    
    # Initialization
    from scipy.stats import cauchy
    np.random.seed(42)
    omega = cauchy.rvs(loc=0, scale=1, size=N)
    theta = np.random.uniform(0, 2*np.pi, N)
    
    # Storage
    time_axis = np.linspace(0, T, steps)
    r_history = np.zeros(steps)
    lambda_history = np.zeros(steps)
    jump_times = []
    
    def compute_order(theta):
        z = np.mean(np.exp(1j * theta))
        return np.abs(z), np.angle(z)
    
    # Simulation Loop
    print("Simulating Market Projection (Ramp Scenario)...")
    for t_idx in range(steps):
        K_current = K_ramp[t_idx]
        
        # 1. Standard Stochastic Kuramoto Step
        r, psi = compute_order(theta)
        drift = omega + K_current * r * np.sin(psi - theta)
        diffusion = sigma * np.random.normal(0, np.sqrt(dt), N)
        theta += drift * dt + diffusion
        
        # 2. Market Projection
        current_lambda = lambda_base + alpha * r
        
        # 3. Poisson Jump Generation
        p_jump = current_lambda * dt
        if np.random.random() < p_jump:
            jump_times.append(time_axis[t_idx])
        
        # Store data
        r_history[t_idx] = r
        lambda_history[t_idx] = current_lambda
    
    results = {
        'time_axis': time_axis,
        'r_history': r_history,
        'lambda_history': lambda_history,
        'jump_times': jump_times,
        'K_ramp': K_ramp
    }
    
    print(f"Total jumps observed: {len(jump_times)}")
    
    # Visualization
    save_path = "Figure3.pdf" if save_figs else None
    plot_market_projection(results, save_path)
    
    print("✓ Block 3 completed\n")


def run_block4(save_figs=True):
    """
    Block 4: Falsifiability Tests.
    """
    print("=" * 60)
    print("BLOCK 4: Falsifiability Check")
    print("=" * 60)
    
    # Configuration
    N = 1000
    T = 50
    dt = 0.05
    steps = int(T / dt)
    time_axis = np.linspace(0, T, steps)
    
    print("Running Counterfactual Checks...")
    
    # Shared random initialization (matches notebook exactly)
    from scipy.stats import cauchy
    np.random.seed(42)
    omega_fixed = cauchy.rvs(loc=0, scale=1, size=N)
    theta_start = np.random.uniform(0, 2*np.pi, N)
    
    def run_scenario(K_val, sigma_val):
        """Inline scenario runner matching notebook implementation."""
        theta = theta_start.copy()
        r_history = []
        
        for _ in range(steps):
            # Compute Order
            z = np.mean(np.exp(1j * theta))
            r, psi = np.abs(z), np.angle(z)
            r_history.append(r)
            
            # Dynamics
            drift = omega_fixed + K_val * r * np.sin(psi - theta)
            diffusion = sigma_val * np.random.normal(0, np.sqrt(dt), N)
            theta += drift * dt + diffusion
        
        return np.array(r_history)
    
    # Scenario 1: Full Model
    r_full = run_scenario(K_val=2.5, sigma_val=0.5)
    
    # Scenario 2: No Coupling
    r_null_coupling = run_scenario(K_val=0.0, sigma_val=0.5)
    
    # Scenario 3: No Noise
    r_null_noise = run_scenario(K_val=2.5, sigma_val=0.0)
    
    # Visualization
    save_path = "Figure4.pdf" if save_figs else None
    plot_counterfactuals(time_axis, r_full, r_null_coupling, r_null_noise, save_path)
    
    print("✓ Block 4 completed\n")


def run_block5(save_figs=True):
    """
    Block 5: Robustness Validation.
    
    Demonstrates:
    - Observability gap in hazard rate
    - Parameter robustness sweep
    """
    print("=" * 60)
    print("BLOCK 5: Robustness Validation")
    print("=" * 60)
    
    # Configuration
    K_safe = 0.5
    K_crit = 1.9
    sigma_base = 0.5
    N_base = 1000
    T = 100
    dt = 0.05
    alpha = 20.0
    lambda_0 = 0.5
    
    # Part A: Observability Check
    print(">>> Running Test A: Observability Gap...")
    
    # Safe scenario
    model_safe = MarketCrashModel(N=N_base, sigma=sigma_base, alpha=alpha, 
                                  lambda_base=lambda_0, seed=42)
    r_safe_series = model_safe.evolve(K_safe, T, dt)
    lam_safe = np.array([model_safe.compute_jump_intensity(r) for r in r_safe_series])
    
    # Critical scenario
    model_crit = MarketCrashModel(N=N_base, sigma=sigma_base, alpha=alpha,
                                  lambda_base=lambda_0, seed=42)
    r_crit_series = model_crit.evolve(K_crit, T, dt)
    lam_crit = np.array([model_crit.compute_jump_intensity(r) for r in r_crit_series])
    
    # Visualization
    save_path = "Figure5.pdf" if save_figs else None
    plot_robustness(lam_safe, lam_crit, save_path)
    
    # Part B: Parameter Robustness Sweep
    print("\n>>> Running Test B: Parameter Robustness Sweep...")
    
    robustness_scenarios = [
        {'sigma': 0.2, 'N': 1000, 'label': 'Low Noise'},
        {'sigma': 0.5, 'N': 1000, 'label': 'Baseline'},
        {'sigma': 0.8, 'N': 1000, 'label': 'High Noise'},
        {'sigma': 0.5, 'N': 500,  'label': 'Small Pop'},
        {'sigma': 0.5, 'N': 2000, 'label': 'Large Pop'},
    ]
    
    print(f"{'Scenario':<15} | {'Sigma':<5} | {'N':<5} | {'Safe AC':<10} | {'Crit AC':<10} | {'Signal Increase':<15}")
    print("-" * 75)
    
    for scen in robustness_scenarios:
        # Safe
        model_s = MarketCrashModel(N=scen['N'], sigma=scen['sigma'], alpha=alpha,
                                   lambda_base=lambda_0, seed=42)
        r_s = model_s.evolve(K_safe, T, dt)
        lam_s = np.array([model_s.compute_jump_intensity(r) for r in r_s])
        diag_s = compute_diagnostics(lam_s)
        ac_safe = diag_s['Autocorrelation']
        
        # Critical
        model_c = MarketCrashModel(N=scen['N'], sigma=scen['sigma'], alpha=alpha,
                                   lambda_base=lambda_0, seed=42)
        r_c = model_c.evolve(K_crit, T, dt)
        lam_c = np.array([model_c.compute_jump_intensity(r) for r in r_c])
        diag_c = compute_diagnostics(lam_c)
        ac_crit = diag_c['Autocorrelation']
        
        # Signal strength
        increase = ((ac_crit - ac_safe) / ac_safe) * 100
        
        print(f"{scen['label']:<15} | {scen['sigma']:<5} | {scen['N']:<5} | {ac_safe:.4f}     | {ac_crit:.4f}     | +{increase:.1f}%")
    
    print("\n(Note: If 'Signal Increase' is positive and significant, the precursors are robust.)")
    print("✓ Block 5 completed\n")


def main():
    """Run all simulations and save figures Figure1.pdf through Figure5.pdf"""
    print("\n" + "="*60)
    print("RUNNING ALL SIMULATION BLOCKS")
    print("="*60 + "\n")
    
    # Run all blocks in order
    run_block1()
    run_block2()
    run_block3()
    run_block4()
    run_block5()
    
    print("\n" + "="*60)
    print("ALL SIMULATIONS COMPLETED")
    print("Figures saved: Figure1.pdf, Figure2.pdf, Figure3.pdf, Figure4.pdf, Figure5.pdf")
    print("="*60)


if __name__ == "__main__":
    main()
