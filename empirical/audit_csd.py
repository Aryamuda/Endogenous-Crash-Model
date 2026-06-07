import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings("ignore")

def main():
    tickers = ["AAPL", "MSFT", "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "AXP",
               "XOM", "CVX", "COP", "SLB", "HAL", "VLO", "OXY", "DVN",
               "JNJ", "PFE", "MRK", "ABT", "BMY", "MDT", "UNH", "HUM", "CI", "AET",
               "WMT", "HD", "TGT", "LOW", "MCD", "YUM", "KO", "PEP", "PM", "MO",
               "GE", "MMM", "HON", "EMR", "CAT", "DE", "BA", "LMT", "RTX", "NOC"]

    # Fetch data
    df = yf.download(tickers, start='2005-01-01', end='2008-09-17', progress=False)
    stocks = df.xs('Close', level=0, axis=1) if isinstance(df.columns, pd.MultiIndex) else df
    stocks = stocks.dropna(axis=1)

    # Log returns
    log_returns = np.log(stocks / stocks.shift(1))

    windows = [30, 60, 90, 126]
    t1_results = []
    
    fig1, axes1 = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig1.suptitle("Test 1: Variance Z-Score Across Different Window Sizes", fontsize=14, fontweight='bold')
    
    for i, w in enumerate(windows):
        corr_matrices = log_returns.rolling(w).corr()
        mean_all = corr_matrices.mean(axis=1).groupby(level=0).mean()
        N = stocks.shape[1]
        r_t = (mean_all * (N**2) - N) / (N * (N - 1))
        
        var_r = r_t.rolling(w).var()
        
        # Standardize using full sample mean and std
        z_var = (var_r - var_r.mean()) / var_r.std()
        
        # Define regimes
        baseline_mask = (r_t.index >= '2005-08-01') & (r_t.index < '2007-02-01')
        warning_mask = (r_t.index >= '2007-06-01') & (r_t.index <= '2008-09-15')
        
        baseline_r = r_t[baseline_mask].mean()
        warning_r = r_t[warning_mask].mean()
        
        peak_var = var_r[warning_mask].max()
        peak_z = z_var[warning_mask].max()
        peak_var_date = var_r[warning_mask].idxmax()
        
        crossings = z_var[(z_var > 2) & warning_mask]
        first_crossing = crossings.index[0] if not crossings.empty else None
        
        t1_results.append({
            'Window': w,
            'Baseline_rt': float(baseline_r),
            'Warning_rt': float(warning_r),
            'Peak_Var': float(peak_var),
            'Peak_Z': float(peak_z),
            'First_2sigma': first_crossing.strftime('%Y-%m-%d') if first_crossing else "None",
            'Peak_Var_Date': peak_var_date.strftime('%Y-%m-%d') if pd.notnull(peak_var_date) else "None"
        })
        
        axes1[i].plot(z_var.index, z_var, label=f'Window = {w} days', color='blue')
        axes1[i].axhline(2, color='red', linestyle=':', label='+2σ Threshold')
        axes1[i].set_ylabel('Z-Score')
        axes1[i].legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('test1_windows.png', dpi=150)
    plt.close()

    # TEST 2: Autocorrelation (window=126)
    w = 126
    corr_matrices = log_returns.rolling(w).corr()
    mean_all = corr_matrices.mean(axis=1).groupby(level=0).mean()
    r_t_126 = (mean_all * (N**2) - N) / (N * (N - 1))
    
    ac1 = r_t_126.rolling(w).apply(lambda x: x.autocorr(lag=1), raw=False)
    z_ac1 = (ac1 - ac1.mean()) / ac1.std()
    
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig2.suptitle("Test 2: Lag-1 Autocorrelation of r(t)", fontsize=14, fontweight='bold')
    axes2[0].plot(ac1.index, ac1, color='blue', label='AC1 of r(t) (126d)')
    axes2[0].set_ylabel('AC1')
    axes2[0].legend(loc='upper left')
    
    axes2[1].plot(z_ac1.index, z_ac1, color='orange', label='Z-Score of AC1')
    axes2[1].axhline(2, color='red', linestyle=':')
    axes2[1].set_ylabel('Z-Score')
    axes2[1].legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('test2_ac1.png', dpi=150)
    plt.close()
    
    base_mask_ac = (ac1.index >= '2006-01-01') & (ac1.index < '2007-02-01')
    warn_mask_ac = (ac1.index >= '2007-06-01') & (ac1.index <= '2008-09-15')
    
    base_ac1 = ac1[base_mask_ac].mean()
    warn_ac1 = ac1[warn_mask_ac].mean()
    peak_ac1 = ac1[warn_mask_ac].max()
    peak_ac1_date = ac1[warn_mask_ac].idxmax()
    
    # TEST 3: Raw vs Z-score (window=126)
    var_126 = r_t_126.rolling(w).var()
    z_var_126 = (var_126 - var_126.mean()) / var_126.std()
    
    fig3, axes3 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig3.suptitle("Test 3: Raw Variance vs Standardized Z-Score", fontsize=14, fontweight='bold')
    axes3[0].plot(var_126.index, var_126, color='purple', label='Raw Variance of r(t)')
    axes3[0].set_ylabel('Raw Variance')
    axes3[0].legend(loc='upper left')
    
    axes3[1].plot(z_var_126.index, z_var_126, color='green', label='Z-Score of Variance (Full Sample)')
    axes3[1].axhline(2, color='red', linestyle=':')
    axes3[1].set_ylabel('Z-Score')
    axes3[1].legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('test3_raw_vs_z.png', dpi=150)
    plt.close()
    
    base_var = var_126[baseline_mask].mean()
    peak_raw = var_126[warning_mask].max()
    pct_increase = ((peak_raw - base_var) / base_var) * 100 if base_var > 0 else 0

    results = {
        'test1': t1_results,
        'test2': {
            'base_ac1': float(base_ac1),
            'warn_ac1': float(warn_ac1),
            'peak_ac1': float(peak_ac1),
            'peak_ac1_date': peak_ac1_date.strftime('%Y-%m-%d') if pd.notnull(peak_ac1_date) else "None"
        },
        'test3': {
            'base_var': float(base_var),
            'peak_raw': float(peak_raw),
            'pct_increase': float(pct_increase)
        }
    }
    
    with open('audit_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
