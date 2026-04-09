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
    
    # London time windows (UTC)
    # Morning: 07:00-09:55
    # Afternoon: 14:00-16:55
    
    df = df.copy()
    
    # Convert to datetime to extract hour
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    
    # Morning window: 7:00-9:55 (hour 7, 8, or 9 before minute 55)
    is_morning = (df['hour'] == 7) | ((df['hour'] == 8) | ((df['hour'] == 9) & (df['minute'] <= 55)))
    
    # Afternoon window: 14:00-16:55 (hour 14, 15, or 16 before minute 55)
    is_afternoon = (df['hour'] == 14) | ((df['hour'] == 15) | ((df['hour'] == 16) & (df['minute'] <= 55)))
    
    # Combined time window
    in_time_window = is_morning | is_afternoon
    
    # FVG Detection (shifted to avoid lookahead)
    # Bullish FVG: current low > high 2 bars ago (gap up)
    # Bearish FVG: current high < low 2 bars ago (gap down)
    
    fvg_high = df['high'].shift(2)
    fvg_low = df['low'].shift(2)
    
    bullish_fvg = (df['low'] > fvg_high).astype(int)
    bearish_fvg = (df['high'] < fvg_low).astype(int)
    
    bullish_fvg_shifted = bullish_fvg.shift(1).fillna(0).astype(int)
    bearish_fvg_shifted = bearish_fvg.shift(1).fillna(0).astype(int)
    
    # Entry conditions: price enters FVG on next bar
    bullish_entry = (bullish_fvg == 1) & (bullish_fvg_shifted == 0)
    bearish_entry = (bearish_fvg == 1) & (bearish_fvg_shifted == 0)
    
    # Apply time window filter
    valid_long_entry = bullish_entry & in_time_window
    valid_short_entry = bearish_entry & in_time_window
    
    # Generate trade signals
    entries = []
    trade_num = 1
    
    for idx in range(len(df)):
        if valid_long_entry.iloc[idx]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': df['time'].iloc[idx],
                'entry_time': df['dt'].iloc[idx].isoformat(),
                'entry_price': df['close'].iloc[idx],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price': 0.0,
                'fvg_high': fvg_high.iloc[idx],
                'fvg_low': fvg_low.iloc[idx]
            })
            trade_num += 1
        elif valid_short_entry.iloc[idx]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': df['time'].iloc[idx],
                'entry_time': df['dt'].iloc[idx].isoformat(),
                'entry_price': df['close'].iloc[idx],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price': 0.0,
                'fvg_high': fvg_high.iloc[idx],
                'fvg_low': fvg_low.iloc[idx]
            })
            trade_num += 1
    
    return entries

# Example usage
df = pd.DataFrame({
    'time': [1609459200, 1609545600, 1609632000],
    'open': [100, 101, 102],
    'high': [105, 106, 107],
    'low': [95, 96, 97],
    'close': [103, 104, 105],
    'volume': [1000, 1100, 1200]
})

entries = generate_entries(df)
print(entries)