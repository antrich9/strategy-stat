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
    trade_num = 0

    # Skip if insufficient bars
    if len(df) < 3:
        return entries

    # Calculate bias based on previous day's close vs prev day high/low
    # Bias calculation uses shift(1) for previous bar values
    prev_close = df['close'].shift(1)
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)

    # Calculate bias: Buy if close > prev_high, Sell if close < prev_low
    bias_bullish = df['close'] > prev_high
    bias_bearish = df['close'] < prev_low

    # Morning window: 07:45 - 09:45 London time
    # Afternoon window: 15:45 - 16:45 London time
    # No Friday trading
    in_trading_window = np.zeros(len(df), dtype=bool)
    
    for i in range(len(df)):
        ts = df['time'].iloc[i]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        is_friday = dt.weekday() == 4  # Friday
        hour = dt.hour
        minute = dt.minute
        
        # Morning window: 07:45 to 09:45
        morning_start = (hour == 7 and minute >= 45) or (hour > 7 and hour < 9)
        morning_end = hour == 9 and minute <= 45
        in_morning = (hour == 7 and minute >= 45) or (hour == 8) or (hour == 9 and minute <= 45)
        
        # Afternoon window: 15:45 to 16:45
        in_afternoon = (hour == 15 and minute >= 45) or (hour == 16 and minute <= 45)
        
        in_trading_window[i] = (in_morning or in_afternoon) and not is_friday

    # Entry conditions
    # Long: bias is bullish AND in trading window
    # Short: bias is bearish AND in trading window
    long_condition = bias_bullish & in_trading_window
    short_condition = bias_bearish & in_trading_window

    # Iterate and generate entries
    for i in range(1, len(df)):
        if np.isnan(df['close'].iloc[i]) or np.isnan(df['high'].iloc[i]) or np.isnan(df['low'].iloc[i]):
            continue
        if np.isnan(prev_close.iloc[i]) or np.isnan(prev_high.iloc[i]) or np.isnan(prev_low.iloc[i]):
            continue

        if long_condition.iloc[i]:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

        if short_condition.iloc[i]:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

    return entries