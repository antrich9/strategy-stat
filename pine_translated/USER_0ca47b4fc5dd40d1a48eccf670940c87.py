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

    # Convert time to datetime for hour extraction
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour

    # Time filter: isValidTradeTime = (hour >= 8 and hour < 12)
    is_valid_time = (df['hour'] >= 8) & (df['hour'] < 12)

    # Helper functions for OB and FVG
    def is_up(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]

    def is_down(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]

    # Calculate conditions for each bar (starting from index 2 to have enough history)
    ob_bullish = pd.Series(False, index=df.index)
    ob_bearish = pd.Series(False, index=df.index)
    fvg_bullish = pd.Series(False, index=df.index)
    fvg_bearish = pd.Series(False, index=df.index)

    for i in range(2, len(df)):
        # Bullish OB: down candle at i-1, up candle at i, close > high of i-1
        ob_bullish.iloc[i] = is_down(i-1) and is_up(i) and df['close'].iloc[i] > df['high'].iloc[i-1]
        # Bearish OB: up candle at i-1, down candle at i, close < low of i-1
        ob_bearish.iloc[i] = is_up(i-1) and is_down(i) and df['close'].iloc[i] < df['low'].iloc[i-1]
        # Bullish FVG: low > high of bar i-2
        fvg_bullish.iloc[i] = df['low'].iloc[i] > df['high'].iloc[i-2]
        # Bearish FVG: high < low of bar i-2
        fvg_bearish.iloc[i] = df['high'].iloc[i] < df['low'].iloc[i-2]

    # Entry conditions
    long_condition = is_valid_time & (ob_bullish | fvg_bullish)
    short_condition = is_valid_time & (ob_bearish | fvg_bearish)

    # Generate long entries
    for i in range(len(df)):
        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
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

    # Generate short entries
    for i in range(len(df)):
        if short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
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