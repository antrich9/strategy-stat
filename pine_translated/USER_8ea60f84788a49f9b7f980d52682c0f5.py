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
    
    # ALMA Settings
    alma_window = 9
    alma_offset = 0.85
    alma_sigma = 6
    
    # FluxWave Settings
    flux_smooth = 1
    flux_length = 50
    flux_offset = 0.85
    flux_sigma = 6
    flux_band_mult = 1.0
    
    # QuickSilver Settings
    qs_period = 20
    qs_deviation = 2.0
    
    # Volume Settings
    vol_length = 50
    vol_high = 150
    vol_normal = 75
    vol_low = 75
    
    # Entry Settings
    entry_lookback = 5
    entry_threshold = 0.5
    qfn_cooldown = 3
    
    # Signal flags
    use_alma_signal = True
    use_flux_signal = True
    use_qs_signal = True
    use_vol_filter = True
    
    # Helper: ALMA calculation
    def alma(src, window, offset, sigma):
        m = (window - 1) * offset
        s = window / sigma
        w = np.exp(-np.square(np.arange(window) - m) / (2 * s * s))
        w = w / w.sum()
        result = np.zeros(len(src))
        for i in range(window - 1, len(src)):
            result[i] = np.sum(w * src[i - window + 1:i + 1])
        return pd.Series(result, index=src.index)
    
    # Helper: Wilder RSI
    def wilder_rsi(src, length):
        delta = src.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Helper: Wilder ATR
    def wilder_atr(high, low, close, length):
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr
    
    # Helper: FluxWave Oscillator (simplified)
    def fluxwave_osc(close, length, smooth, alma_offset_val, alma_sigma_val):
        # Price momentum with ALMA smoothing
        src = close.ewm(span=smooth, adjust=False).mean()
        # Calculate wave using ALMA of momentum
        momentum = src.pct_change(length)
        wave = alma(momentum, length, alma_offset_val, alma_sigma_val)
        # Normalize with standard deviation bands
        std = momentum.rolling(length).std()
        upper = wave + flux_band_mult * std
        lower = wave - flux_band_mult * std
        return wave, upper, lower
    
    # Helper: QuickSilver Bands
    def quicksilver_bands(close, period, deviation):
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = sma + deviation * std
        lower = sma - deviation * std
        mid = sma
        return upper, lower, mid
    
    # Helper: Volume Energy
    def volume_energy(volume, length, high_thresh, normal_thresh, low_thresh):
        # Normalized volume using percentile rank
        vol_ma = volume.rolling(length).mean()
        vol_std = volume.rolling(length).std()
        vol_z = (volume - vol_ma) / vol_std
        # Energy level based on z-score
        energy = vol_z * 50 + 100  # Centered around 100
        return energy
    
    # Calculate indicators
    close = df['close'].copy()
    high = df['high'].copy()
    low = df['low'].copy()
    volume = df['volume'].copy()
    
    # ALMA Baseline
    alma_val = alma(close, alma_window, alma_offset, alma_sigma)
    
    # ALMA Trend Direction
    alma_trend = alma_val.diff()
    
    # FluxWave
    flux_wave, flux_upper, flux_lower = fluxwave_osc(close, flux_length, flux_smooth, flux_offset, flux_sigma)
    
    # FluxWave Signal
    flux_signal = pd.Series(0, index=close.index)
    flux_above_zero = flux_wave > 0
    flux_prev_above = flux_wave.shift(1) <= 0
    flux_below_zero = flux_wave < 0
    flux_prev_below = flux_wave.shift(1) >= 0
    flux_cross_up = (flux_above_zero & flux_prev_above).astype(int)
    flux_cross_down = (flux_below_zero & flux_prev_below).astype(int)
    flux_signal = flux_cross_up - flux_cross_down
    
    # QuickSilver
    qs_upper, qs_lower, qs_mid = quicksilver_bands(close, qs_period, qs_deviation)
    
    # QuickSilver breakout signal
    qs_breakout_up = close > qs_upper
    qs_breakout_down = close < qs_lower
    qs_prev_below = close.shift(1) <= qs_upper
    qs_prev_above = close.shift(1) >= qs_lower
    qs_cross_up = qs_breakout_up & qs_prev_below
    qs_cross_down = qs_breakout_down & qs_prev_above
    
    # Volume Energy
    vol_energy = volume_energy(volume, vol_length, vol_high, vol_normal, vol_low)
    vol_confirm = vol_energy > vol_normal
    
    # ALMA Crossover signals
    alma_price_cross_up = (close > alma_val) & (close.shift(1) <= alma_val.shift(1))
    alma_price_cross_down = (close < alma_val) & (close.shift(1) >= alma_val.shift(1))
    
    # Build entry conditions
    long_cond = pd.Series(True, index=close.index)
    short_cond = pd.Series(True, index=close.index)
    
    if use_alma_signal:
        long_cond = long_cond & alma_price_cross_up
        short_cond = short_cond & alma_price_cross_down
    
    if use_flux_signal:
        long_cond = long_cond & (flux_signal > 0)
        short_cond = short_cond & (flux_signal < 0)
    
    if use_qs_signal:
        long_cond = long_cond & qs_cross_up
        short_cond = short_cond & qs_cross_down
    
    if use_vol_filter:
        long_cond = long_cond & vol_confirm
        short_cond = short_cond & vol_confirm
    
    # Skip NaN bars for indicators
    valid_start = max(alma_window, flux_length, qs_period, vol_length) + 1
    
    # Generate entries
    entries = []
    trade_num = 1
    last_entry_bar = -qfn_cooldown - 1
    
    for i in range(valid_start, len(df)):
        # Check cooldown
        if i - last_entry_bar <= qfn_cooldown:
            continue
        
        # Long entry
        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
            last_entry_bar = i
            continue
        
        # Short entry
        if short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
            last_entry_bar = i
    
    return entries