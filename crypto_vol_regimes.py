import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ccxt
from datetime import datetime
from arch import arch_model
from hmmlearn.hmm import GaussianHMM

# Constants
START_DATE = "2024-10-01"
END_DATE = "2024-10-12"
COLOR_HIGH = '#FF4500'  # Orange-red for high volatility
COLOR_LOW = '#00CED1'  # Cyan for low volatility

def fetch_high_freq_data(ticker):
    """Fetch 1-minute BTC/USDT data from Binance."""
    exchange = ccxt.binance()
    start_ts = exchange.parse8601(f"{START_DATE}T00:00:00Z")
    end_ts = exchange.parse8601(f"{END_DATE}T00:00:00Z")
    ohlcv = exchange.fetch_ohlcv(ticker, timeframe='1m', since=start_ts, limit=10000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[df['timestamp'] <= pd.to_datetime(END_DATE)]
    df.set_index('timestamp', inplace=True)
    returns = df['close'].pct_change().dropna() * 100  # Percentage returns with DatetimeIndex
    return df, returns

def fit_garch_midas(returns):
    """Fit GARCH-MIDAS model to separate short- and long-term volatility."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Returns must have a DatetimeIndex")
    
    daily_vol = returns.resample('1D').std().dropna() * np.sqrt(252)  # Annualized
    model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', 
                       dist='normal', rescale=True)
    result = model.fit(disp='off')
    
    short_term_vol = result.conditional_volatility
    long_term_vol = daily_vol.reindex(returns.index, method='ffill')
    
    return short_term_vol, long_term_vol

def detect_regimes(volatility):
    """Detect volatility regimes using HMM."""
    hmm = GaussianHMM(n_components=2, covariance_type='diag', n_iter=100)
    hmm.fit(volatility.values.reshape(-1, 1))
    regimes = hmm.predict(volatility.values.reshape(-1, 1))
    return regimes, hmm.means_.flatten(), hmm  # Return hmm object

def plot_volatility_heatmap(df, short_term_vol, long_term_vol, regimes, hmm, ticker):
    """Create a quant-style volatility clustering heatmap with regime overlay."""
    times = df.index
    vol_combined = np.log(short_term_vol + long_term_vol)
    print(f"times length: {len(times)}")
    print(f"vol_combined length: {len(vol_combined)}")
    
    # Align times with vol_combined
    times = times[1:]  # Match vol_combined length after dropna
    if len(times) != len(vol_combined):
        raise ValueError(f"times ({len(times)}) and vol_combined ({len(vol_combined)}) must have same length")
    print(f"Adjusted times length: {len(times)}")
    
    # Define bins
    n_time_bins = 264  # Approx 11 days * 24 hours
    n_vol_bins = 50
    time_edges = np.linspace(times[0].timestamp(), times[-1].timestamp(), n_time_bins + 1)
    vol_edges = np.linspace(vol_combined.min(), vol_combined.max(), n_vol_bins + 1)
    print(f"time_edges length: {len(time_edges)}")
    print(f"vol_edges length: {len(vol_edges)}")
    
    # Create heatmap data
    heatmap_data, x_edges, y_edges = np.histogram2d(
        times.astype('int64') / 10**9,
        vol_combined,
        bins=[n_time_bins, n_vol_bins],
        range=[[times[0].timestamp(), times[-1].timestamp()], [vol_edges[0], vol_edges[-1]]]
    )
    print(f"Raw heatmap_data shape: {heatmap_data.shape}")
    
    # Transpose for Plotly (y, x)
    heatmap_data = np.log1p(heatmap_data.T)  # Shape: (50, 264)
    print(f"Transposed heatmap_data shape: {heatmap_data.shape}")
    
    # Bin centers
    x_bins = 0.5 * (x_edges[:-1] + x_edges[1:])  # 264 points
    y_bins = 0.5 * (y_edges[:-1] + y_edges[1:])  # 50 points
    x_bins_datetime = pd.to_datetime(x_bins, unit='s')
    print(f"x_bins length: {len(x_bins)}")
    print(f"y_bins length: {len(y_bins)}")
    
    # Dimension check
    n_y, n_x = heatmap_data.shape
    if n_x != len(x_bins) or n_y != len(y_bins):
        raise ValueError(f"Dimension mismatch: z shape {heatmap_data.shape}, x length {len(x_bins)}, y length {len(y_bins)}")
    
    # Regime transitions
    regime_changes = np.diff(regimes) != 0
    transition_times = times[1:][regime_changes]
    regime_labels = ['Low Vol', 'High Vol'] if hmm.means_[0] < hmm.means_[1] else ['High Vol', 'Low Vol']
    
    fig = go.Figure()
    
    # Heatmap
    fig.add_trace(go.Heatmap(
        z=heatmap_data,
        x=x_bins_datetime,
        y=y_bins,
        colorscale=[[0, COLOR_LOW], [1, COLOR_HIGH]],
        opacity=0.9,
        showscale=True,
        colorbar=dict(
            title=dict(text='Log Density', side='right'),
            tickfont=dict(color='white')
        )
    ))
    
    # Regime transitions
    for t in transition_times:
        fig.add_vline(x=t, line=dict(color='white', width=1, dash='dash'))
    
    # Annotate regimes
    regime_start_times = [times[0]] + transition_times.tolist() + [times[-1]]
    for i in range(len(regime_start_times) - 1):
        mid_point = regime_start_times[i] + (regime_start_times[i+1] - regime_start_times[i]) / 2
        regime_idx = regimes[times.get_indexer([regime_start_times[i]], method='nearest')[0]]  # Corrected get_loc
        fig.add_annotation(
            x=mid_point, y=y_bins.max(),
            text=regime_labels[regime_idx],
            showarrow=False,
            font=dict(color='white', size=12),
            bgcolor='rgba(0,0,0,0.5)'
        )
    
    fig.update_layout(
        title=dict(text=f'{ticker} Volatility Clustering (GARCH-MIDAS + HMM)', font_color='white', x=0.5),
        xaxis_title=dict(text='Time', font_color='white'),
        yaxis_title=dict(text='Log Volatility', font_color='white'),
        plot_bgcolor='rgb(40,40,40)',
        paper_bgcolor='rgb(40,40,40)',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickangle=45, color='white'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.show()

    # Additional plot: Volatility time series with regimes
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=times,
        y=vol_combined,
        mode='lines',
        line=dict(color='white', width=1),
        name='Log Volatility'
    ))
    for t in transition_times:
        fig2.add_vline(x=t, line=dict(color='red', width=1, dash='dash'))
    fig2.update_layout(
        title=dict(text=f'{ticker} Volatility with Regime Transitions', font_color='white', x=0.5),
        xaxis_title=dict(text='Time', font_color='white'),
        yaxis_title=dict(text='Log Volatility', font_color='white'),
        plot_bgcolor='rgb(40,40,40)',
        paper_bgcolor='rgb(40,40,40)',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickangle=45, color='white'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    fig2.show()

if __name__ == "__main__":
    ticker = 'BTC/USDT'
    try:
        df, returns = fetch_high_freq_data(ticker)
        short_term_vol, long_term_vol = fit_garch_midas(returns)
        regimes, means, hmm = detect_regimes(short_term_vol)
        print(f"Volatility Regimes Detected - Means: {means}")
        plot_volatility_heatmap(df, short_term_vol, long_term_vol, regimes, hmm, ticker)
        
    except Exception as e:
        print(f"Error: {str(e)}")