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
    
    # Parse timestamps to datetime for time-based filtering
    df = df.copy()
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    df['weekday'] = df['dt'].dt.weekday
    
    # London session time windows (UTC)
    # Morning: 08:00 - 09:55
    # Afternoon: 14:00 - 16:55
    london_morning = ((df['hour'] == 8) & (df['minute'] >= 0)) | \
                     ((df['hour'] == 9) & (df['minute'] <= 55))
    london_afternoon = ((df['hour'] == 14) & (df['minute'] >= 0)) | \
                       ((df['hour'] == 15) & (df['minute'] <= 55)) | \
                       ((df['hour'] == 16) & (df['minute'] <= 55))
    in_time_window = london_morning | london_afternoon
    
    # Calculate EMAs for trend (5-min data used, but on daily df just use close)
    ema_fast = df['close'].ewm(span=10, adjust=False).mean()
    ema_slow = df['close'].ewm(span=50, adjust=False).mean()
    
    # Trend: Fast EMA > Slow EMA = Bullish, < = Bearish
    trend_bullish = ema_fast > ema_slow
    trend_bearish = ema_fast < ema_slow
    
    # FVG Detection: Bullish FVG = low >= high[2] and low[2] >= high (3-bar pattern)
    # Bearish FVG = high <= low[2] and high[2] <= low
    # For bullish: current bar's low is above previous bar's high (gap up)
    # For bearish: current bar's high is below previous bar's low (gap down)
    
    high_1 = df['high'].shift(1)
    low_1 = df['low'].shift(1)
    high_2 = df['high'].shift(2)
    low_2 = df['low'].shift(2)
    high_3 = df['high'].shift(3)
    low_3 = df['low'].shift(3)
    
    # Bullish FVG: current low >= high of 2 bars ago (gap up filled or tested)
    bullish_fvg = (df['low'] >= high_2) & (df['low'].shift(1) < high_3)
    # Bearish FVG: current high <= low of 2 bars ago (gap down filled or tested)
    bearish_fvg = (df['high'] <= low_2) & (df['high'].shift(1) > low_3)
    
    # Swing detection for entry confirmation
    # Swing high: high[1] < high[2] and high[3] < high[2]
    # Swing low: low[1] > low[2] and low[3] > low[2]
    swing_high = (df['high'].shift(1) < df['high'].shift(2)) & (df['high'].shift(3) < df['high'].shift(2))
    swing_low = (df['low'].shift(1) > df['low'].shift(2)) & (df['low'].shift(3) > df['low'].shift(2))
    
    # Long entry: Bullish FVG detected in bullish trend within time window
    # Short entry: Bearish FVG detected in bearish trend within time window
    long_condition = bullish_fvg & trend_bullish & in_time_window
    short_condition = bearish_fvg & trend_bearish & in_time_window
    
    # Generate entries
    for i in range(len(df)):
        if i < 5:  # Need at least 5 bars for all indicators
            continue
        
        if long_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries