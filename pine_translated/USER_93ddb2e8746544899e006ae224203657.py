import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.

    Returns list of dicts:
    [{'trade_num': int, 'direction': 'long' or 'short',
      'entry_ts': int, 'entry_time': str,
      'entry_price_guess': float,
      'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
      'raw_price_a': float, 'raw_price_b': float}]
    """
    entries = []
    trade_num = 1
    
    # Parameters (default values from Pine Script inputs)
    alma_window = 9
    alma_offset = 0.85
    alma_sigma = 6.0
    flux_smooth = 1
    flux_length = 50
    flux_offset = 0.85
    flux_sigma = 6
    flux_band_mult = 1.0
    qs_period = 20
    qs_deviation = 2.0
    vol_length = 50
    vol_high = 150
    vol_normal = 75
    vol_low = 75
    entry_threshold = 0.5
    entry_lookback = 5
    qfn_cooldown = 3
    entry_mode = "Combined"
    use_alma_signal = True
    use_flux_signal = True
    use_qs_signal = True
    use_vol_filter = True
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # ALMA Implementation
    def alma_calc(series, window, offset, sigma):
        m = np.floor(offset * (window - 1))
        s = sigma * (window - 1) / 2
        weights = np.exp(-np.power(np.arange(window) - m, 2) / (2 * s * s))
        weights = weights / weights.sum()
        return pd.Series(series).rolling(window).apply(lambda x: np.sum(weights * x), raw=True)
    
    # ALMA Baseline
    alma_baseline = alma_calc(close, alma_window, alma_offset, alma_sigma)
    
    # QuickSilver Bands
    qs_ma = close.rolling(qs_period).mean()
    qs_std = close.rolling(qs_period).std()
    qs_upper = qs_ma + qs_deviation * qs_std
    qs_lower = qs_ma - qs_deviation * qs_std
    
    # FluxWave Oscillator
    tsf = close.ewm(span=flux_length, adjust=False).mean()
    src_flux = close
    flux_osc_raw = src_flux - tsf
    flux_osc = alma_calc(flux_osc_raw, flux_smooth + 1, flux_offset, flux_sigma)
    flux_upper_band = flux_band_mult * flux_osc.rolling(flux_length).std()
    flux_lower_band = -flux_band_mult * flux_osc.rolling(flux_length).std()
    
    # Volume Energy Filter
    vol_ma = volume.rolling(vol_length).mean()
    vol_std = volume.rolling(vol_length).std()
    vol_energy = ((volume - vol_ma) / (vol_std + 1e-9)) * 100
    
    # Build condition series
    alma_long_cond = (close > alma_baseline) & (close.shift(1) <= alma_baseline.shift(1))
    alma_short_cond = (close < alma_baseline) & (close.shift(1) >= alma_baseline.shift(1))
    
    flux_long_cond = (flux_osc > flux_upper_band) & (flux_osc.shift(1) <= flux_upper_band.shift(1))
    flux_short_cond = (flux_osc < flux_lower_band) & (flux_osc.shift(1) >= flux_lower_band.shift(1))
    
    qs_long_cond = (close > qs_upper) & (close.shift(1) <= qs_upper.shift(1))
    qs_short_cond = (close < qs_lower) & (close.shift(1) >= qs_lower.shift(1))
    
    vol_cond = vol_energy > vol_normal
    
    # Combined entry conditions
    long_cond = pd.Series(False, index=df.index)
    short_cond = pd.Series(False, index=df.index)
    
    if entry_mode == "Combined":
        long_cond = alma_long_cond & flux_long_cond & qs_long_cond & (vol_cond if use_vol_filter else True)
        short_cond = alma_short_cond & flux_short_cond & qs_short_cond & (vol_cond if use_vol_filter else True)
    elif entry_mode == "ALMA":
        long_cond = alma_long_cond
        short_cond = alma_short_cond
    elif entry_mode == "FluxWave":
        long_cond = flux_long_cond
        short_cond = flux_short_cond
    elif entry_mode == "QuickSilver":
        long_cond = qs_long_cond
        short_cond = qs_short_cond
    
    # Cooldown tracking
    last_trade_idx = -qfn_cooldown - 1
    
    for i in range(entry_lookback, len(df)):
        if pd.isna(alma_baseline.iloc[i]) or pd.isna(flux_osc.iloc[i]) or pd.isna(qs_upper.iloc[i]):
            continue
        
        if i <= last_trade_idx + qfn_cooldown:
            continue
        
        direction = None
        if use_alma_signal and entry_mode == "Combined":
            if long_cond.iloc[i]:
                direction = 'long'
            elif short_cond.iloc[i]:
                direction = 'short'
        elif use_flux_signal and entry_mode in ["Combined", "FluxWave"]:
            if flux_long_cond.iloc[i] or (use_alma_signal and alma_long_cond.iloc[i]):
                direction = 'long'
            elif flux_short_cond.iloc[i] or (use_alma_signal and alma_short_cond.iloc[i]):
                direction = 'short'
        elif use_qs_signal and entry_mode in ["Combined", "QuickSilver"]:
            if qs_long_cond.iloc[i]:
                direction = 'long'
            elif qs_short_cond.iloc[i]:
                direction = 'short'
        elif entry_mode == "ALMA":
            if alma_long_cond.iloc[i]:
                direction = 'long'
            elif alma_short_cond.iloc[i]:
                direction = 'short'
        
        if direction:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
            last_trade_idx = i
    
    return entries