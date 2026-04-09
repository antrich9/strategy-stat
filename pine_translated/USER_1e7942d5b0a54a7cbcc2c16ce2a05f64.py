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
    
    # Constants from Pine Script
    barhtf_bar_ratio = 7  # Default ratio for Weekly vs Daily (240/60 min in Pine = 4, but using 7 for W/D setup)
    
    # Calculate number of bars to look back for HTF FVG detection
    n = barhtf_bar_ratio
    
    # Get the series
    highs = df['high']
    lows = df['low']
    opens = df['open']
    closes = df['close']
    bartimes = df['time']
    
    # Skip if not enough data
    if len(df) < n + 2:
        return entries
    
    # Detect HTF Bullish FVG: low > high of bar n+1 bars ago
    # Pine: htf_bartimes > htf_bartimes[1] and htf_lows > htf_highs[(barhtf / bar) + 1]
    # We need htf_bartimes != htf_bartimes[1] which means a new HTF candle started
    # For simplicity, we check every n bars (when HTF candle changes)
    
    # Create HTF candle identification
    # A new HTF candle starts every n bars
    htf_candle_starts = pd.Series([False] * len(df))
    htf_candle_starts.iloc[n:] = True  # Simplified: assume new HTF candle every n bars
    
    # Bullish FVG: current low > high from n+1 bars ago
    bullish_fvg = (lows > highs.shift(n + 1)) & htf_candle_starts
    
    # Bearish FVG: current high < low from n+1 bars ago
    bearish_fvg = (highs < lows.shift(n + 1)) & htf_candle_starts
    
    # Generate long entries
    for i in range(n + 1, len(df)):
        if bullish_fvg.iloc[i] and not np.isnan(highs.iloc[i]):
            ts = int(bartimes.iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': closes.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': closes.iloc[i],
                'raw_price_b': closes.iloc[i]
            }
            entries.append(entry)
            trade_num += 1
    
    # Generate short entries
    for i in range(n + 1, len(df)):
        if bearish_fvg.iloc[i] and not np.isnan(lows.iloc[i]):
            ts = int(bartimes.iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': closes.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': closes.iloc[i],
                'raw_price_b': closes.iloc[i]
            }
            entries.append(entry)
            trade_num += 1
    
    return entries