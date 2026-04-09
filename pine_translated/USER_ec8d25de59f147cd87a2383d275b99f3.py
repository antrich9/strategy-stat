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
    
    # Input parameters (defaults from Pine Script)
    threshold_per = 0.5  # Default fvgTH
    auto = False  # Default auto
    
    high_series = df['high']
    low_series = df['low']
    close_series = df['close']
    time_series = df['time']
    
    # Detect new day
    dt_times = pd.to_datetime(time_series, unit='s')
    day_change = dt_times.dt.date != dt_times.dt.date.shift(1)
    new_day = day_change.fillna(False)
    
    # Calculate Previous Day High and Low
    pd_high_arr = pd.Series(np.nan, index=df.index)
    pd_low_arr = pd.Series(np.nan, index=df.index)
    temp_high = np.nan
    temp_low = np.nan
    
    for i in df.index:
        if new_day.iloc[i]:
            pd_high_arr.iloc[i] = temp_high
            pd_low_arr.iloc[i] = temp_low
            temp_high = high_series.iloc[i]
            temp_low = low_series.iloc[i]
        else:
            temp_high = high_series.iloc[i] if np.isnan(temp_high) else max(temp_high, high_series.iloc[i])
            temp_low = low_series.iloc[i] if np.isnan(temp_low) else min(temp_low, low_series.iloc[i])
    
    # Sweep detection
    swept_high = False
    swept_low = False
    sweep_high_signal = pd.Series(False, index=df.index)
    sweep_low_signal = pd.Series(False, index=df.index)
    
    for i in df.index:
        if new_day.iloc[i]:
            swept_high = False
            swept_low = False
        
        if not swept_high and high_series.iloc[i] > pd_high_arr.iloc[i]:
            sweep_high_signal.iloc[i] = True
            swept_high = True
        
        if not swept_low and low_series.iloc[i] < pd_low_arr.iloc[i]:
            sweep_low_signal.iloc[i] = True
            swept_low = True
        
        if swept_high and swept_low:
            swept_high = False
            swept_low = False
    
    # Calculate threshold (same as Pine ta.cum logic)
    if auto:
        bar_returns = (high_series - low_series) / low_series
        threshold = bar_returns.cumsum() / np.arange(1, len(df) + 1)
    else:
        threshold = threshold_per / 100
    
    # Bullish FVG: low > high[2] and close[1] > high[2] and (low - high[2]) / high[2] > threshold
    high_2 = high_series.shift(2)
    close_1 = close_series.shift(1)
    bull_fvg = (low_series > high_2) & (close_1 > high_2) & ((low_series - high_2) / high_2 > threshold)
    bull_fvg = bull_fvg.fillna(False)
    
    # Bearish FVG: high < low[2] and close[1] < low[2] and (low[2] - high) / high > threshold
    low_2 = low_series.shift(2)
    bear_fvg = (high_series < low_2) & (close_1 < low_2) & ((low_2 - high_series) / high_series > threshold)
    bear_fvg = bear_fvg.fillna(False)
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in df.index:
        ts = int(time_series.iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = float(close_series.iloc[i])
        
        # Long entry conditions: sweepHighNow OR bull_fvg
        if sweep_high_signal.iloc[i] or bull_fvg.iloc[i]:
            entries.append({
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
            })
            trade_num += 1
        
        # Short entry conditions: sweepLowNow OR bear_fvg
        if sweep_low_signal.iloc[i] or bear_fvg.iloc[i]:
            entries.append({
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
            })
            trade_num += 1
    
    return entries