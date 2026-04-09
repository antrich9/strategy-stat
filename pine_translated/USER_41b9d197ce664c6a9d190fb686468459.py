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
    
    # Convert time to datetime for time window filtering
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    
    # London time windows (in minutes from midnight)
    # Morning: 08:00 to 09:55 (480 to 595)
    # Afternoon: 14:00 to 16:55 (840 to 1015)
    london_morning_start = 8 * 60
    london_morning_end = 9 * 60 + 55
    london_afternoon_start = 14 * 60
    london_afternoon_end = 16 * 60 + 55
    
    is_london_morning = (df['time_minutes'] >= london_morning_start) & (df['time_minutes'] < london_morning_end)
    is_london_afternoon = (df['time_minutes'] >= london_afternoon_start) & (df['time_minutes'] < london_afternoon_end)
    is_within_time_window = is_london_morning | is_london_afternoon
    
    # FVG Detection conditions using close for pattern detection
    # Bullish FVG (Bottom_ImbXBAway): high[2] >= open[1] and low[0] <= close[1] and close[0] < high[1]
    bullish_fvg = (
        (df['high'].shift(2) >= df['open'].shift(1)) &
        (df['low'] <= df['close'].shift(1)) &
        (df['close'] < df['high'].shift(1))
    )
    
    # Bearish FVG (Top_ImbXBway): low[2] <= open[1] and high[0] >= close[1] and close[0] > low[1]
    bearish_fvg = (
        (df['low'].shift(2) <= df['open'].shift(1)) &
        (df['high'] >= df['close'].shift(1)) &
        (df['close'] > df['low'].shift(1))
    )
    
    # Entry conditions: price enters the FVG zone
    # Bullish entry: low < high[1] (entering the bullish FVG zone)
    bullish_entry_cond = (df['low'] < df['high'].shift(1)) & bullish_fvg & is_within_time_window
    
    # Bearish entry: high > low[1] (entering the bearish FVG zone)
    bearish_entry_cond = (df['high'] > df['low'].shift(1)) & bearish_fvg & is_within_time_window
    
    # Filter out NaN values (first 2 bars won't have valid FVG conditions)
    bullish_entry_cond = bullish_entry_cond.fillna(False)
    bearish_entry_cond = bearish_entry_cond.fillna(False)
    
    # Ensure we skip bars with NaN in required shift columns
    valid_bars = ~df['high'].shift(1).isna() & ~df['low'].shift(1).isna() & ~df['close'].shift(1).isna()
    bullish_entry_cond = bullish_entry_cond & valid_bars
    bearish_entry_cond = bearish_entry_cond & valid_bars
    
    entries = []
    trade_num = 1
    
    # Iterate through all bars
    for i in range(len(df)):
        # Check bullish entries
        if bullish_entry_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
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
            trade_num += 1
        
        # Check bearish entries
        if bearish_entry_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
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
            trade_num += 1
    
    return entries