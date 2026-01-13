import numpy as np
import matplotlib.pyplot as plt

# Color-blind safe palette (Okabe-Ito palette)
CB_COLORS = {
    'blue': '#0173B2',      # Blue
    'orange': '#DE8F05',    # Orange
    'green': '#029E73',     # Green (bluish)
    'red': '#CC78BC',       # Reddish purple (magenta instead of red)
    'cyan': '#56B4E9',      # Sky blue
    'purple': '#949494',    # Gray
    'black': '#000000',     # Black
    'yellow': '#ECE133'     # Yellow
}

# Line styles for additional distinction
LINE_STYLES = ['-', '--', '-.', ':']
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>']


def plot_phase_transition(K_values, r_steady, K_c=2.0, save_path=None):
    """
    Plot phase transition curve.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(K_values, r_steady, 'o-', color=CB_COLORS['black'], 
             linewidth=2, markersize=6, label='Simulation')
    plt.axvline(x=K_c, color=CB_COLORS['red'], linestyle='--', 
                linewidth=2.5, label=f'Theoretical $K_c={K_c}$')
    plt.title("Phase Transition: Order vs Coupling", fontsize=14, fontweight='bold')
    plt.xlabel("Coupling Strength $K$", fontsize=12)
    plt.ylabel("Order Parameter $r_{\infty}$", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    
    plt.close()


def plot_time_evolution(time_axis, time_series, labels, colors, save_path=None):
    """
    Plot time evolution for multiple regimes.
    """
    plt.figure(figsize=(10, 6))
    
    # Color-blind safe colors
    cb_colors_list = [CB_COLORS['blue'], CB_COLORS['orange'], CB_COLORS['green']]
    
    for i, series in enumerate(time_series):
        plt.plot(time_axis, series, label=labels[i], 
                color=cb_colors_list[i], linewidth=2.5, 
                linestyle=LINE_STYLES[i], alpha=0.9)
    
    plt.title("Time Evolution of Synchronization", fontsize=14, fontweight='bold')
    plt.xlabel("Time $t$", fontsize=12)
    plt.ylabel("Order Parameter $r(t)$", fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    
    plt.close()


def plot_block1(K_values, r_steady, time_axis, time_series, labels, colors, save_path=None):
    """
    Combined plot for Block 1: Phase transition + time evolution.
    """
    plt.figure(figsize=(14, 6))
    
    # Left Panel: Phase Transition
    plt.subplot(1, 2, 1)
    plt.plot(K_values, r_steady, 'o-', color=CB_COLORS['black'], 
             linewidth=2, markersize=6, label='Simulation')
    plt.axvline(x=2.0, color=CB_COLORS['red'], linestyle='--', 
                linewidth=2.5, label='Theoretical $K_c=2$')
    plt.title("Phase Transition: Order vs Coupling", fontsize=13, fontweight='bold')
    plt.xlabel("Coupling Strength $K$", fontsize=11)
    plt.ylabel("Order Parameter $r_{\infty}$", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Right Panel: Time Evolution
    plt.subplot(1, 2, 2)
    cb_colors_list = [CB_COLORS['blue'], CB_COLORS['orange'], CB_COLORS['green']]
    for i, series in enumerate(time_series):
        plt.plot(time_axis, series, label=labels[i], 
                color=cb_colors_list[i], linewidth=2.5, 
                linestyle=LINE_STYLES[i], alpha=0.9)
    
    plt.title("Time Evolution of Synchronization", fontsize=13, fontweight='bold')
    plt.xlabel("Time $t$", fontsize=11)
    plt.ylabel("Order Parameter $r(t)$", fontsize=11)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    
    plt.close()


def plot_stochastic_precursors(results, scenarios, steps, save_path=None):
    """
    Plot stochastic precursor analysis (Block 2).
    """
    plt.figure(figsize=(12, 6))
    
    # Plot R(t) Time Series
    plt.subplot(2, 1, 1)
    plt.plot(results['Safe'][int(steps*0.2):], 
             label=f'Safe (K={scenarios["Safe"]})', 
             color=CB_COLORS['blue'], linewidth=1.5, alpha=0.9, linestyle='-')
    plt.plot(results['Critical'][int(steps*0.2):], 
             label=f'Critical (K={scenarios["Critical"]})', 
             color=CB_COLORS['orange'], linewidth=1.5, alpha=0.9, linestyle='--')
    
    plt.title("Order Parameter Fluctuations: Safe vs. Critical", fontsize=13, fontweight='bold')
    plt.ylabel("$r(t)$", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot Histogram of Fluctuations
    plt.subplot(2, 2, 3)
    plt.hist(results['Safe'][int(steps*0.2):], bins=30, alpha=0.7, 
             label='Safe', color=CB_COLORS['blue'], edgecolor='black', linewidth=0.5)
    plt.hist(results['Critical'][int(steps*0.2):], bins=30, 
             color=CB_COLORS['orange'], alpha=0.7, label='Critical', 
             edgecolor='black', linewidth=0.5)
    plt.title("Distribution of Order $r(t)$", fontsize=11)
    plt.legend(fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    
    plt.close()


def plot_market_projection(results_dict, save_path=None):
    """
    Plot market projection with 3 layers (Block 3).
    """
    time_axis = results_dict['time_axis']
    r_history = results_dict['r_history']
    lambda_history = results_dict['lambda_history']
    jump_times = results_dict['jump_times']
    T = time_axis[-1]
    
    plt.figure(figsize=(12, 8))
    
    # Panel 1: Synchronization
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, r_history, color=CB_COLORS['blue'], 
             linewidth=2, label='Order Parameter $r(t)$')
    plt.axvline(x=T/2, color=CB_COLORS['red'], linestyle='--', 
                linewidth=2.5, alpha=0.7, label=r'Theoretical $K_c \approx 2$')
    plt.ylabel("Synchronization $r(t)$", fontsize=11)
    plt.title("Layer 1: Endogenous Synchronization Building Up", fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Panel 2: Hazard Rate
    plt.subplot(3, 1, 2)
    plt.plot(time_axis, lambda_history, color=CB_COLORS['red'], 
             linewidth=2, linestyle='-.', label=r'Jump Intensity $\lambda(t)$')
    plt.ylabel(r"Hazard Rate $\lambda(t)$", fontsize=11)
    plt.title("Layer 2: Market Instability (Slaved to Sync)", fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Panel 3: Market Jumps
    plt.subplot(3, 1, 3)
    if jump_times:
        plt.vlines(jump_times, 0, 1, color=CB_COLORS['black'], 
                  linewidth=1.5, label='Market Jumps')
    plt.xlabel("Time (Proxy for Increasing Coupling K)", fontsize=11)
    plt.yticks([])
    plt.title("Layer 3: Observed Market Jumps (The Crash)", fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    
    plt.close()


def plot_counterfactuals(time_axis, r_full, r_null_coupling, r_null_noise, save_path=None):
    """
    Plot counterfactual scenarios (Block 4).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, r_full, label='Full Model (Sync + Fluctuations)', 
            color=CB_COLORS['green'], linewidth=2.5, linestyle='-')
    plt.plot(time_axis, r_null_coupling, label='Null Coupling (K=0) -> No Sync', 
            color=CB_COLORS['purple'], linewidth=2, linestyle=':', alpha=0.8)
    plt.plot(time_axis, r_null_noise, label='Null Noise (sigma=0) -> No Fluctuations', 
            color=CB_COLORS['red'], linewidth=2.5, linestyle='--')
    
    plt.title("Falsifiability Check: Counterfactual Scenarios", fontsize=14, fontweight='bold')
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Order Parameter r(t)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    
    plt.close()


def plot_robustness(lam_safe, lam_crit, save_path=None):
    """
    Plot robustness check for observability gap (Block 5).
    """
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(lam_safe[200:], label='Safe (K=0.5)', 
            color=CB_COLORS['blue'], linewidth=1.5, linestyle='-', alpha=0.9)
    plt.plot(lam_crit[200:], label='Critical (K=1.9)', 
            color=CB_COLORS['orange'], linewidth=1.5, linestyle='--', alpha=0.9)
    plt.title("Observable Hazard Rate $\lambda(t)$", fontsize=12, fontweight='bold')
    plt.xlabel("Time Steps", fontsize=11)
    plt.ylabel("Intensity $\lambda(t)$", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(lam_safe[200:], bins=30, alpha=0.7, label='Safe', 
            color=CB_COLORS['blue'], edgecolor='black', linewidth=0.5)
    plt.hist(lam_crit[200:], bins=30, color=CB_COLORS['orange'], 
            alpha=0.7, label='Critical', edgecolor='black', linewidth=0.5)
    plt.title("Distribution of Hazard Rate", fontsize=12, fontweight='bold')
    plt.xlabel("$\lambda(t)$", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    
    plt.close()
