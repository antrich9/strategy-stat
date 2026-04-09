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
    
    # Parameters from Pine Script inputs (using defaults/balanced mode)
    alma_window = 9
    alma_offset = 0.85
    alma_sigma = 6.0
    flux_length = 50
    flux_smooth = 1
    flux_offset = 0.85
    flux_sigma = 6
    qs_period = 20
    qs_deviation = 2.0
    vol_length = 50
    entry_lookback = 5
    entry_threshold = 0.5
    qfn_cooldown = 3
    
    # Signal mode settings
    use_alma_signal = True
    use_flux_signal = True
    use_qs_signal = True
    use_vol_filter = True
    
    close = df['close'].copy()
    high = df['high'].copy()
    low = df['low'].copy()
    volume = df['volume'].copy()
    
    # ===================== ALMA Calculation =====================
    def alma(src, window, offset, sigma):
        m = offset * (window - 1)
        s = sigma * (window - 1) / 6
        w = np.exp(-np.square(np.arange(window) - m) / (2 * s * s))
        alma_vals = np.zeros(len(src))
        for i in range(window - 1, len(src)):
            alma_vals[i] = np.sum(w * src[i - window + 1:i + 1]) / np.sum(w)
        return pd.Series(alma_vals, index=src.index)
    
    alma_values = alma(close, alma_window, alma_offset, alma_sigma)
    alma_diff = close - alma_values
    alma_diff_prev = alma_diff.shift(1)
    
    # ===================== FluxWave Calculation =====================
    def fluxwave_alma(src, window, offset, sigma):
        m = offset * (window - 1)
        s = sigma * (window - 1) / 6
        w = np.exp(-np.square(np.arange(window) - m) / (2 * s * s))
        fw_vals = np.zeros(len(src))
        for i in range(window - 1, len(src)):
            fw_vals[i] = np.sum(w * src[i - window + 1:i + 1]) / np.sum(w)
        return pd.Series(fw_vals, index=src.index)
    
    # FluxWave uses rate of change of ALMA
    flux_src = fluxwave_alma(close, flux_length, flux_offset, flux_sigma)
    flux_change = flux_src.diff(flux_smooth)
    flux_change_smooth = fluxwave_alma(flux_change.fillna(0), flux_length, flux_offset, flux_sigma)
    
    # FluxWave signal line
    flux_signal = fluxwave_alma(flux_change_smooth, flux_length, flux_offset, flux_sigma)
    
    # FluxWave normalized
    flux_max = flux_change_smooth.rolling(flux_length).max()
    flux_min = flux_change_smooth.rolling(flux_length).min()
    flux_range = flux_max - flux_min
    flux_range = flux_range.replace(0, np.nan)
    fluxwave_norm = (flux_change_smooth - flux_min) / flux_range
    
    # ===================== QuickSilver Bands =====================
    qs_ma = close.rolling(qs_period).mean()
    qs_std = close.rolling(qs_period).std()
    qs_upper = qs_ma + qs_deviation * qs_std
    qs_lower = qs_ma - qs_deviation * qs_std
    
    # QuickSilver breakout signal
    qs_bullish_breakout = close > qs_upper
    qs_bearish_breakout = close < qs_lower
    
    # ===================== Volume Energy Calculation =====================
    vol_ma = volume.rolling(vol_length).mean()
    vol_std = volume.rolling(vol_length).std()
    vol_energy = (volume - vol_ma) / (vol_std + 1e-10)
    vol_energy_norm = (vol_energy - vol_energy.rolling(vol_length).min()) / \
                      (vol_energy.rolling(vol_length).max() - vol_energy.rolling(vol_length).min() + 1e-10)
    
    # ===================== Entry Signal Construction =====================
    # ALMA crossover signals
    alma_bullish_cross = (alma_diff > 0) & (alma_diff_prev <= 0)
    alma_bearish_cross = (alma_diff < 0) & (alma_diff_prev >= 0)
    
    # FluxWave overbought/oversold signals
    flux_oversold = fluxwave_norm < 0.2
    flux_overbought = fluxwave_norm > 0.8
    
    # Combined entry conditions
    long_signal = pd.Series(False, index=df.index)
    short_signal = pd.Series(False, index=df.index)
    
    for i in range(vol_length + alma_window + flux_length + qs_period, len(df)):
        if pd.isna(alma_values.iloc[i]) or pd.isna(fluxwave_norm.iloc[i]) or pd.isna(qs_ma.iloc[i]):
            continue
        
        # Count confirmations
        long_confirmations = 0
        short_confirmations = 0
        
        # ALMA confirmation
        if use_alma_signal and alma_bullish_cross.iloc[i]:
            long_confirmations += 1
        if use_alma_signal and alma_bearish_cross.iloc[i]:
            short_confirmations += 1
        
        # FluxWave confirmation
        if use_flux_signal:
            if flux_oversold.iloc[i]:
                long_confirmations += 1
            if flux_overbought.iloc[i]:
                short_confirmations += 1
        
        # QuickSilver confirmation
        if use_qs_signal:
            if qs_bullish_breakout.iloc[i]:
                long_confirmations += 1
            if qs_bearish_breakout.iloc[i]:
                short_confirmations += 1
        
        # Volume confirmation
        if use_vol_filter:
            if vol_energy_norm.iloc[i] > 0.3:
                long_confirmations += 1
            if vol_energy_norm.iloc[i] > 0.3:
                short_confirmations += 1
        
        # Check entry threshold
        max_confirmations = sum([use_alma_signal, use_flux_signal, use_qs_signal, use_vol_filter])
        strength = max_confirmations / max_confirmations if max_confirmations > 0 else 0
        
        if long_confirmations >= 2 and strength >= entry_threshold:
            long_signal.iloc[i] = True
        if short_confirmations >= 2 and strength >= entry_threshold:
            short_signal.iloc[i] = True
    
    # ===================== Generate Entry List =====================
    entries = []
    trade_num = 1
    last_entry_bar = -qfn_cooldown - 1
    
    for i in range(len(df)):
        # Check cooldown
        if i - last_entry_bar < qfn_cooldown:
            continue
        
        entry_price = df['close'].iloc[i]
        
        # Long entry
        if long_signal.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
            last_entry_bar = i
        
        # Short entry
        if short_signal.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
            last_entry_bar = i
    
    return entries