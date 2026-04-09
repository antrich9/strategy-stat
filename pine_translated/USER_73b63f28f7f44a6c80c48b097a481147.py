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
    lookback_bars = 12
    threshold = 0.0
    
    # Bullish FVG detection
    bull_fvg_arr = (df['low'] > df['high'].shift(2)) & (df['close'].shift(1) > df['high'].shift(2))
    
    # Bearish FVG detection
    bear_fvg_arr = (df['high'] < df['low'].shift(2)) & (df['close'].shift(1) < df['low'].shift(2))
    
    # Track bars since last bullish FVG
    bull_since_arr = np.zeros(len(df))
    last_bull_idx = -1
    for i in range(len(df)):
        if bull_fvg_arr.iloc[i]:
            last_bull_idx = i
        bull_since_arr[i] = i - last_bull_idx if last_bull_idx != -1 else np.nan
    
    # Track bars since last bearish FVG
    bear_since_arr = np.zeros(len(df))
    last_bear_idx = -1
    for i in range(len(df)):
        if bear_fvg_arr.iloc[i]:
            last_bear_idx = i
        bear_since_arr[i] = i - last_bear_idx if last_bear_idx != -1 else np.nan
    
    # Bull result calculation
    bull_result_arr = np.zeros(len(df), dtype=bool)
    bull_cond_1_arr = bull_fvg_arr & (bull_since_arr <= lookback_bars)
    bull_since_series = pd.Series(bull_since_arr)
    
    for i in range(len(df)):
        if bull_cond_1_arr.iloc[i]:
            idx = int(bull_since_series.iloc[i])
            combined_low = max(df['high'].iloc[idx], df['high'].iloc[2]) if idx >= 0 else np.nan
            combined_high = min(df['low'].iloc[idx + 2], df['low'].iloc[i]) if idx + 2 < len(df) else np.nan
            bull_result_arr[i] = bull_cond_1_arr.iloc[i] and not np.isnan(combined_low) and not np.isnan(combined_high) and (combined_high - combined_low >= threshold)
        else:
            bull_result_arr[i] = False
    
    # Bear result calculation
    bear_result_arr = np.zeros(len(df), dtype=bool)
    bear_cond_1_arr = bear_fvg_arr & (bear_since_arr <= lookback_bars)
    bear_since_series = pd.Series(bear_since_arr)
    
    for i in range(len(df)):
        if bear_cond_1_arr.iloc[i]:
            idx = int(bear_since_series.iloc[i])
            combined_low = max(df['high'].iloc[idx + 2], df['high'].iloc[i]) if idx + 2 < len(df) else np.nan
            combined_high = min(df['low'].iloc[idx], df['low'].iloc[2]) if idx >= 0 else np.nan
            bear_result_arr[i] = bear_cond_1_arr.iloc[i] and not np.isnan(combined_low) and not np.isnan(combined_high) and (combined_high - combined_low >= threshold)
        else:
            bear_result_arr[i] = False
    
    # Entry generation
    entries = []
    trade_num = 1
    entry_emitted = np.zeros(len(df), dtype=bool)
    last_label_bull = False
    
    for i in range(len(df)):
        if bull_result_arr[i] and last_label_bull and not entry_emitted[i]:
            ts = df['time'].iloc[i]
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
            entry_emitted[i] = True
        
        if bull_fvg_arr.iloc[i]:
            last_label_bull = True
    
    return entries