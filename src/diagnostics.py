import numpy as np
import pandas as pd


def calculate_variance(series, window=200, burn_in_fraction=0.2):
    """
    Calculate rolling variance on time series.
    """
    cut = int(len(series) * burn_in_fraction)
    clean_series = pd.Series(series[cut:])
    
    rolling_var = clean_series.rolling(window=window).var()
    mean_var = rolling_var.mean()
    
    return rolling_var, mean_var


def calculate_autocorrelation(series, lag=1, window=200, burn_in_fraction=0.2):
    """
    Calculate rolling lag-1 autocorrelation.
    """
    cut = int(len(series) * burn_in_fraction)
    clean_series = pd.Series(series[cut:])
    
    rolling_ac = clean_series.rolling(window=window).apply(
        lambda x: x.autocorr(lag=lag), raw=False
    )
    mean_ac = rolling_ac.mean()
    
    return rolling_ac, mean_ac


def compute_diagnostics(series, window=200, burn_in_fraction=0.2):
    """
    Compute multiple diagnostic metrics for a time series.
    """
    _, var = calculate_variance(series, window, burn_in_fraction)
    _, ac = calculate_autocorrelation(series, 1, window, burn_in_fraction)
    
    diagnostics = {
        'Variance': var,
        'Autocorrelation': ac
    }
    
    return diagnostics


def compare_scenarios(scenarios_dict, window=200, burn_in_fraction=0.2):
    """
    Compare diagnostic metrics across multiple scenarios.
    """
    results = {}
    
    for name, series in scenarios_dict.items():
        results[name] = compute_diagnostics(series, window, burn_in_fraction)
    
    # Convert to DataFrame for easy viewing
    df = pd.DataFrame(results).T
    
    return df


def detect_early_warning(series, threshold_percentile=90, window=200):
    """
    Detect early warning signals based on variance threshold.
    """
    rolling_var, _ = calculate_variance(series, window, burn_in_fraction=0.0)
    threshold = np.percentile(rolling_var.dropna(), threshold_percentile)
    
    # Find first crossing
    above_threshold = rolling_var > threshold
    warning_indices = np.where(above_threshold)[0]
    
    if len(warning_indices) > 0:
        return True, warning_indices[0]
    else:
        return False, None


def signal_to_noise_ratio(safe_series, critical_series, metric='Variance'):
    """
    Calculate signal-to-noise ratio between safe and critical regimes.
    """
    safe_diag = compute_diagnostics(safe_series)
    crit_diag = compute_diagnostics(critical_series)
    
    safe_val = safe_diag[metric]
    crit_val = crit_diag[metric]
    
    snr = ((crit_val - safe_val) / safe_val) * 100
    
    return snr


def robustness_table(scenarios, K_safe, K_crit, base_params):
    """
    Generate robustness table for parameter sweep.
    """
    from .model import StochasticKuramotoModel
    
    results = []
    
    for scen in scenarios:
        # Run safe scenario
        model_safe = StochasticKuramotoModel(
            N=scen['N'],
            sigma=scen['sigma'],
            seed=42
        )
        r_safe = model_safe.evolve(K_safe, base_params['T'], base_params['dt'])
        ac_safe, _ = calculate_autocorrelation(r_safe)
        
        # Run critical scenario
        model_crit = StochasticKuramotoModel(
            N=scen['N'],
            sigma=scen['sigma'],
            seed=42
        )
        r_crit = model_crit.evolve(K_crit, base_params['T'], base_params['dt'])
        ac_crit, _ = calculate_autocorrelation(r_crit)
        
        # Calculate signal increase
        increase = ((ac_crit - ac_safe) / ac_safe) * 100
        
        results.append({
            'Scenario': scen['label'],
            'Sigma': scen['sigma'],
            'N': scen['N'],
            'Safe AC': ac_safe,
            'Crit AC': ac_crit,
            'Signal Increase': f'+{increase:.1f}%'
        })
    
    return pd.DataFrame(results)
