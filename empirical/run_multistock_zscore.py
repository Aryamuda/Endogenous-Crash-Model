import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

CB_COLORS = {
    'blue': '#0173B2', 'orange': '#DE8F05', 'green': '#029E73',
    'red': '#CC78BC', 'cyan': '#56B4E9', 'purple': '#949494',
    'black': '#000000', 'yellow': '#ECE133'
}

def main():
    tickers = ["AAPL", "MSFT", "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "AXP",
               "XOM", "CVX", "COP", "SLB", "HAL", "VLO", "OXY", "DVN",
               "JNJ", "PFE", "MRK", "ABT", "BMY", "MDT", "UNH", "HUM", "CI", "AET",
               "WMT", "HD", "TGT", "LOW", "MCD", "YUM", "KO", "PEP", "PM", "MO",
               "GE", "MMM", "HON", "EMR", "CAT", "DE", "BA", "LMT", "RTX", "NOC"]
    
    print("Downloading data (2005 - Sep 2008)...")
    # Fetch data up to right after Lehman collapse
    df = yf.download(tickers + ['^GSPC'], start='2005-01-01', end='2008-09-17', progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        close_df = df.xs('Close', level=0, axis=1)
    else:
        close_df = df
        
    sp500 = close_df['^GSPC']
    stocks = close_df.drop(columns=['^GSPC']).dropna(axis=1)
    
    returns = stocks.pct_change()
    
    window = 126
    corr_matrices = returns.rolling(window).corr()
    mean_all = corr_matrices.mean(axis=1).groupby(level=0).mean()
    N = stocks.shape[1]
    
    # r(t) proxy
    r_t = (mean_all * (N**2) - N) / (N * (N - 1))
    
    # Rolling variance
    var_r = r_t.rolling(window).var()
    
    # Calculate Z-Score based on the "Safe" baseline (2005-08 to 2007-02)
    baseline = var_r[(var_r.index >= '2005-08-01') & (var_r.index < '2007-02-01')]
    mu = baseline.mean()
    sigma = baseline.std()
    
    z_score = (var_r - mu) / sigma
    
    print("Plotting...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    lehman_date = pd.to_datetime('2008-09-15')
    
    # Focus X-axis strictly before/at Lehman
    start_date = pd.to_datetime('2005-08-01')
    end_date = pd.to_datetime('2008-09-15')
    
    # Panel 1: Price
    axes[0].plot(sp500.index, sp500, color=CB_COLORS['black'], linewidth=1.5, label='S&P 500 Index')
    axes[0].axvline(x=lehman_date, color=CB_COLORS['red'], linestyle='--', linewidth=2, label='Lehman Collapse')
    axes[0].set_ylabel('Price', fontsize=11)
    axes[0].set_title('Empirical Network Synchronization & CSD Precursors (Pre-Crash Zoom)', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: r(t)
    axes[1].plot(r_t.index, r_t, color=CB_COLORS['blue'], linewidth=1.5, label=r'Mean Pairwise Correlation $r(t)$')
    axes[1].axvline(x=lehman_date, color=CB_COLORS['red'], linestyle='--', linewidth=2)
    axes[1].set_ylabel('Correlation $r(t)$', fontsize=11)
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Z-Score of Variance
    axes[2].plot(z_score.index, z_score, color=CB_COLORS['orange'], linewidth=1.5, label=r'Z-Score of $r(t)$ Variance')
    
    # Highlight Z > 2
    axes[2].axhline(2, color=CB_COLORS['red'], linestyle=':', linewidth=2, label='+2$\sigma$ Early Warning Threshold')
    axes[2].fill_between(z_score.index, 2, z_score, where=(z_score > 2), color=CB_COLORS['red'], alpha=0.3, interpolate=True)
    
    axes[2].axvline(x=lehman_date, color=CB_COLORS['red'], linestyle='--', linewidth=2)
    axes[2].set_ylabel('Variance Z-Score', fontsize=11)
    axes[2].set_xlabel('Date', fontsize=11)
    axes[2].legend(loc='upper left', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # Set X-limits to zoom in
    axes[2].set_xlim(start_date, end_date)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    
    save_path = 'Figure6.pdf'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

if __name__ == '__main__':
    main()
