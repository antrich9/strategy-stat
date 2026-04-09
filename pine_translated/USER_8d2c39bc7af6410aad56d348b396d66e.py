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
    
    # Parameters (using defaults from Pine Script)
    emaFastLen = 8
    emaSlowLen = 21
    voShortLen = 5
    voLongLen = 10
    voThreshold = 0.0
    atrLength = 14
    baselineType = "EMA Cloud"
    volumeType = "Volume Oscillator"
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # === Baseline Calculations ===
    
    # EMA Cloud
    ema_fast = close.ewm(span=emaFastLen, adjust=False).mean()
    ema_slow = close.ewm(span=emaSlowLen, adjust=False).mean()
    
    # === Volume Filter Calculations ===
    
    # Volume Oscillator
    vo_short_ema = volume.ewm(span=voShortLen, adjust=False).mean()
    vo_long_ema = volume.ewm(span=voLongLen, adjust=False).mean()
    volume_oscillator = ((vo_short_ema - vo_long_ema) / vo_long_ema) * 100
    
    # === Wilder RSI Implementation ===
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # === Wilder ATR Implementation ===
    def wilder_atr(high_series, low_series, close_series, length):
        tr1 = high_series - low_series
        tr2 = (high_series - close_series.shift(1)).abs()
        tr3 = (low_series - close_series.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr
    
    atr = wilder_atr(high, low, close, atrLength)
    
    # === Entry Logic ===
    # Baseline: EMA Cloud - fast above slow for long, below for short
    baseline_long = ema_fast > ema_slow
    baseline_short = ema_fast < ema_slow
    
    # Volume: Volume Oscillator - above threshold for long, below for short
    vol_long = volume_oscillator > voThreshold
    vol_short = volume_oscillator < -voThreshold
    
    # Combined entry conditions
    long_condition = baseline_long & vol_long
    short_condition = baseline_short & vol_short
    
    # Crossover detection for entries
    # For long: ema_fast crosses above ema_slow
    # For short: ema_fast crosses below ema_slow
    
    # Manual crossover detection
    ema_fast_above = ema_fast > ema_slow
    ema_fast_below = ema_fast < ema_slow
    
    # Long entry: ema_fast crosses above ema_slow AND volume confirms
    long_entry = (
        (ema_fast > ema_slow) & 
        (ema_fast.shift(1) <= ema_slow.shift(1)) &
        (volume_oscillator > voThreshold)
    )
    
    # Short entry: ema_fast crosses below ema_slow AND volume confirms
    short_entry = (
        (ema_fast < ema_slow) & 
        (ema_fast.shift(1) >= ema_slow.shift(1)) &
        (volume_oscillator < -voThreshold)
    )
    
    entries = []
    trade_num = 1
    
    n = len(df)
    
    for i in range(n):
        # Skip if ATR is NaN (early bars)
        if pd.isna(atr.iloc[i]):
            continue
        
        entry_price = close.iloc[i]
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        if long_entry.iloc[i]:
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            }
            entries.append(entry)
            trade_num += 1
        elif short_entry.iloc[i]:
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            }
            entries.append(entry)
            trade_num += 1
    
    return entries