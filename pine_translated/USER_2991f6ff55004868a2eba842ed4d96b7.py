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
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_price = df['open']
    
    # ALMA parameters
    alma_window = 9
    alma_offset = 0.85
    alma_sigma = 6.0
    
    # FluxWave parameters
    flux_length = 50
    flux_smooth = 1
    flux_offset = 0.85
    flux_sigma = 6
    flux_band_mult = 1.0
    
    # QuickSilver parameters
    qs_period = 20
    qs_deviation = 2.0
    
    # Volume parameters
    vol_length = 50
    vol_high = 150
    vol_normal = 75
    
    # Entry settings
    entry_threshold = 0.5
    entry_lookback = 5
    
    # Input flags
    use_alma_signal = True
    use_flux_signal = True
    use_qs_signal = True
    use_vol_filter = True
    entry_mode = "Combined"
    
    # Calculate ALMA
    def alma(src, window, offset, sigma):
        m = (offset * (window - 1))
        s = (window - 1) / sigma
        w = np.exp(-(np.arange(window) - m)**2 / (2 * s**2))
        w = w / w.sum()
        result = np.zeros(len(src))
        for i in range(window - 1, len(src)):
            result[i] = np.sum(w * src.iloc[i - window + 1:i + 1].values)
        return pd.Series(result, index=src.index)
    
    # Calculate FluxWave
    def fluxwave(src, length, smooth, alma_offset, alma_sigma):
        momentum = src.diff(length)
        alma_mom = alma(momentum, smooth, alma_offset, alma_sigma)
        return alma_mom
    
    # Calculate QuickSilver Bands
    def quicksilver_bands(src, period, deviation):
        sma = src.rolling(period).mean()
        std = src.rolling(period).std()
        upper = sma + (std * deviation)
        lower = sma - (std * deviation)
        return sma, upper, lower
    
    # Calculate Volume Energy
    def volume_energy(vol, length):
        typical = (high + low + close) / 3
        raw_energy = typical * vol
        ema_energy = raw_energy.ewm(span=length, adjust=False).mean()
        ema_vol = volume.ewm(span=length, adjust=False).mean()
        return (ema_energy / ema_vol.replace(0, np.nan)) * 1000
    
    # Compute indicators
    alma_val = alma(close, alma_window, alma_offset, alma_sigma)
    alma_trend = alma(open_price, alma_window, alma_offset, alma_sigma)
    
    fluxwave_val = fluxwave(close, flux_length, flux_smooth, flux_offset, flux_sigma)
    
    qs_sma, qs_upper, qs_lower = quicksilver_bands(close, qs_period, qs_deviation)
    
    vol_energy = volume_energy(volume, vol_length)
    
    # Long signal condition
    alma_long = (close > alma_val) & (alma_trend > alma_val)
    flux_long = fluxwave_val > 0
    qs_long = close > qs_upper
    
    # Short signal condition
    alma_short = (close < alma_val) & (alma_trend < alma_val)
    flux_short = fluxwave_val < 0
    qs_short = close < qs_lower
    
    # Volume filter
    vol_confirm = vol_energy > vol_normal
    
    # Build combined conditions based on entry mode
    if entry_mode == "Combined":
        long_cond = alma_long & flux_long & qs_long
        short_cond = alma_short & flux_short & qs_short
    elif entry_mode == "ALMA":
        long_cond = alma_long
        short_cond = alma_short
    elif entry_mode == "FluxWave":
        long_cond = flux_long
        short_cond = flux_short
    elif entry_mode == "QuickSilver":
        long_cond = qs_long
        short_cond = qs_short
    else:
        long_cond = alma_long & flux_long
        short_cond = alma_short & flux_short
    
    # Apply volume filter
    if use_vol_filter:
        long_cond = long_cond & vol_confirm
        short_cond = short_cond & vol_confirm
    
    # Apply individual signal flags
    if not use_alma_signal:
        long_cond = long_cond & ~alma_long | alma_long
        short_cond = short_cond & ~alma_short | alma_short
    if not use_flux_signal:
        long_cond = long_cond & ~flux_long | flux_long
        short_cond = short_cond & ~flux_short | flux_short
    if not use_qs_signal:
        long_cond = long_cond & ~qs_long | qs_long
        short_cond = short_cond & ~qs_short | qs_short
    
    # Identify crossover and crossunder for ALMA
    alma_crossover = (alma_val > alma_trend) & (alma_val.shift(1) <= alma_trend.shift(1))
    alma_crossunder = (alma_val < alma_trend) & (alma_val.shift(1) >= alma_trend.shift(1))
    
    # Build final signal conditions
    long_entry_cond = long_cond & alma_crossover
    short_entry_cond = short_cond & alma_crossunder
    
    # Ensure first valid bar has data
    first_valid = max(alma_window, flux_length, qs_period, vol_length)
    if len(df) <= first_valid:
        return []
    
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if i < first_valid:
            continue
        
        if pd.isna(alma_val.iloc[i]) or pd.isna(fluxwave_val.iloc[i]):
            continue
        
        entry_price = close.iloc[i]
        ts = int(df['time'].iloc[i])
        
        if long_entry_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_entry_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries