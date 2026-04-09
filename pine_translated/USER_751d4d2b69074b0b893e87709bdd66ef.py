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
    
    if len(df) < 6:
        return []
    
    # Ensure required columns exist
    required_cols = ['time', 'open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            return []
    
    # Convert timestamp to datetime for time window filtering
    df['dt'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_convert('Europe/London')
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    
    # London trading windows (morning and afternoon sessions)
    morning_window = ((df['hour'] == 8) | 
                       ((df['hour'] == 9) & (df['minute'] <= 55)))
    afternoon_window = ((df['hour'] == 14) | 
                        ((df['hour'] == 15) & (df['minute'] <= 59)) | 
                        ((df['hour'] == 16) & (df['minute'] <= 55)))
    in_trading_window = morning_window | afternoon_window
    
    # Swing detection (15m style: main bar vs immediate neighbors)
    swing_high = ((df['high'] > df['high'].shift(1)) & 
                  (df['high'] > df['high'].shift(3)) & 
                  (df['high'].shift(2) >= df['high'].shift(1)) & 
                  (df['high'].shift(2) >= df['high'].shift(3)))
    
    swing_low = ((df['low'] < df['low'].shift(1)) & 
                 (df['low'] < df['low'].shift(3)) & 
                 (df['low'].shift(2) <= df['low'].shift(1)) & 
                 (df['low'].shift(2) <= df['low'].shift(3)))
    
    # FVG detection (15m style from Pine script)
    # Bullish FVG: current bar creates upward gap vs 2 bars back
    bullish_fvg = (df['low'] >= df['high'].shift(2)) & (df['low'].shift(1) < df['high'].shift(2))
    
    # Bearish FVG: current bar creates downward gap vs 2 bars back
    bearish_fvg = (df['high'] <= df['low'].shift(2)) & (df['high'].shift(1) > df['low'].shift(2))
    
    # FVG entry conditions (when price enters FVG zone)
    # Bullish entry: price enters bullish FVG (from above) -> bearish signal
    # Bearish entry: price enters bearish FVG (from below) -> bullish signal
    
    # Long entry: high crosses above bearish FVG top (high of bar 2 back)
    long_entry = (df['high'] > df['high'].shift(2)) & (df['high'].shift(1) <= df['high'].shift(2)) & bearish_fvg.shift(1).fillna(False)
    
    # Short entry: low crosses below bullish FVG bottom (low of bar 2 back)
    short_entry = (df['low'] < df['low'].shift(2)) & (df['low'].shift(1) >= df['low'].shift(2)) & bullish_fvg.shift(1).fillna(False)
    
    # Filter by trading window
    long_entry = long_entry & in_trading_window
    short_entry = short_entry & in_trading_window
    
    # Generate entries
    trade_num = 1
    entries = []
    
    for i in range(len(df)):
        if long_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries