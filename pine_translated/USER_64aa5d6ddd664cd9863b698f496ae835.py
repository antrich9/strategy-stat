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
    results = []
    trade_num = 1
    
    # Helper function for Wilder RSI
    def wilder_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Helper function for Wilder ATR
    def wilder_atr(high, low, close, period):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        return atr
    
    # Helper function for Wilder smoothing
    def wilder_smooth(series, period):
        return series.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # Extract columns for readability
    open_col = df['open'].values
    high_col = df['high'].values
    low_col = df['low'].values
    close_col = df['close'].values
    volume_col = df['volume'].values
    time_col = df['time'].values
    
    n = len(df)
    
    # Convert to pandas Series for indicator calculation
    open_s = df['open']
    high_s = df['high']
    low_s = df['low']
    close_s = df['close']
    volume_s = df['volume']
    
    # RSI (14 period Wilder)
    rsi = wilder_rsi(close_s, 14)
    
    # ATR (14 period Wilder)
    atr = wilder_atr(high_s, low_s, close_s, 14)
    
    # Volume SMA (9 period)
    vol_sma = volume_s.rolling(9).mean()
    
    # Trend SMA (54 period)
    loc = close_s.rolling(54).mean()
    
    # FVG conditions (bfvg, sfvg)
    # Bullish FVG: low > high[2] (gap up from 2 bars ago)
    # Bearish FVG: high < low[2] (gap down from 2 bars ago)
    bfvg = low_s > high_s.shift(2)
    sfvg = high_s < low_s.shift(2)
    
    # Volume filter: volume[1] > sma(volume, 9) * 1.5
    vol_filt = volume_s.shift(1) > vol_sma * 1.5
    
    # ATR filter: (low - high[2] > atr/1.5) or (low[2] - high > atr/1.5)
    atr_val = atr / 1.5
    atr_filt = ((low_s - high_s.shift(2) > atr_val) | (low_s.shift(2) - high_s > atr_val))
    
    # Trend filter
    loc2 = loc > loc.shift(1)
    locfiltb = loc2  # For long entries
    locfilts = ~loc2  # For short entries
    
    # Combined FVG conditions
    long_condition = bfvg & vol_filt & atr_filt & locfiltb
    short_condition = sfvg & vol_filt & atr_filt & locfilts
    
    # RSI filter: RSI < 30 for long, RSI > 70 for short
    rsi_long_filt = rsi < 30
    rsi_short_filt = rsi > 70
    
    # Apply RSI filter to conditions
    long_cond = long_condition & rsi_long_filt
    short_cond = short_condition & rsi_short_filt
    
    # Time window filtering (London time: 7:45-9:45 and 14:45-16:45)
    def check_london_window(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        # Morning window: 7:45 to 9:45
        in_morning = (hour == 7 and minute >= 45) or (hour == 8) or (hour == 9 and minute <= 44)
        # Afternoon window: 14:45 to 16:45
        in_afternoon = (hour == 14 and minute >= 45) or (hour == 15) or (hour == 16 and minute <= 44)
        return in_morning or in_afternoon
    
    time_window = pd.Series([check_london_window(ts) for ts in time_col], index=df.index)
    
    # Apply time window
    long_cond = long_cond & time_window
    short_cond = short_cond & time_window
    
    # RSI crossovers for confirmation (RSI crossing above 30 for longs, below 70 for shorts)
    rsi_cross_above_30 = (rsi > 30) & (rsi.shift(1) <= 30)
    rsi_cross_below_70 = (rsi < 70) & (rsi.shift(1) >= 70)
    
    # Final entry conditions
    long_entry_cond = long_cond & rsi_cross_above_30
    short_entry_cond = short_cond & rsi_cross_below_70
    
    # Iterate and collect entries
    for i in range(n):
        # Skip bars where indicators might be NaN (need at least 54 bars for SMA, etc.)
        if i < 55:
            continue
        
        # Skip if any required indicator is NaN
        if pd.isna(loc.iloc[i]) or pd.isna(rsi.iloc[i]) or pd.isna(atr.iloc[i]):
            continue
        if pd.isna(vol_sma.iloc[i]):
            continue
        
        direction = None
        
        if long_entry_cond.iloc[i]:
            direction = 'long'
        elif short_entry_cond.iloc[i]:
            direction = 'short'
        
        if direction:
            ts = int(time_col[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(close_col[i])
            
            results.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return results