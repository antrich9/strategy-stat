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
    
    # Ensure we have enough data
    if len(df) < 20:
        return entries
    
    # Create timezone-aware datetime column for filtering
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Calculate daily high/low for swing detection
    df['daily_high'] = df['high'].rolling(window=1).max()
    df['daily_low'] = df['low'].rolling(window=1).min()
    
    # Get daily data shifted for swing detection
    dailyHigh11 = df['high'].rolling(window=1).max().shift(1)
    dailyLow11 = df['low'].rolling(window=1).min().shift(1)
    dailyHigh21 = dailyHigh11.shift(1)
    dailyLow21 = dailyLow11.shift(1)
    dailyHigh22 = dailyHigh11.shift(2)
    dailyLow22 = dailyLow11.shift(2)
    
    # Swing detection conditions
    is_swing_high11 = (dailyHigh21 < dailyHigh22) & (dailyHigh11.shift(3) < dailyHigh22) & (dailyHigh11.shift(4) < dailyHigh22)
    is_swing_low11 = (dailyLow21 > dailyLow22) & (dailyLow11.shift(3) > dailyLow22) & (dailyLow11.shift(4) > dailyLow22)
    
    # Track last swing type
    last_swing_type = "none"
    last_swing_high11 = np.nan
    last_swing_low11 = np.nan
    
    # Calculate 4H data manually by resampling
    df['datetime_index'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df_4h = df.set_index('datetime_index').resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    
    # Calculate 4H indicators
    if len(df_4h) < 5:
        return entries
    
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    close_4h = df_4h['close']
    high_4h_1 = high_4h.shift(1)
    low_4h_1 = low_4h.shift(1)
    high_4h_2 = high_4h.shift(2)
    low_4h_2 = low_4h.shift(2)
    close_4h_1 = close_4h.shift(1)
    
    # FVG conditions using historical 4H data
    bfvg_condition = low_4h > high_4h_2
    sfvg_condition = high_4h < low_4h_2
    
    # Detect new day for sweep logic
    df['date'] = df['datetime'].dt.date
    df['new_day'] = df['date'].diff().fillna(False).astype(bool)
    
    # Calculate previous day high/low for sweep detection
    df['pdHigh'] = df['high'].cummax().where(df['new_day'], np.nan)
    df['pdLow'] = df['low'].cummin().where(df['new_day'], np.nan)
    
    # Fill forward pdHigh and pdLow within each day
    df['pdHigh'] = df.groupby('date')['pdHigh'].transform(lambda x: x.ffill())
    df['pdLow'] = df.groupby('date')['pdLow'].transform(lambda x: x.ffill())
    
    # Shift to get previous day's values
    df['prev_day_high'] = df['pdHigh'].shift(1)
    df['prev_day_low'] = df['pdLow'].shift(1)
    
    # Track sweep status per day
    df['sweptHigh'] = False
    df['sweptLow'] = False
    
    # Calculate trading windows (London time: 07:00-11:45 and 14:00-14:45)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    
    # Window 1: 07:00-11:45 (420-705 minutes)
    window1_start = 7 * 60
    window1_end = 11 * 60 + 45
    is_within_window1 = (df['time_minutes'] >= window1_start) & (df['time_minutes'] < window1_end)
    
    # Window 2: 14:00-14:45 (840-885 minutes)
    window2_start = 14 * 60
    window2_end = 14 * 60 + 45
    is_within_window2 = (df['time_minutes'] >= window2_start) & (df['time_minutes'] < window2_end)
    
    in_trading_window = is_within_window1 | is_within_window2
    
    # Create bar confirmation (simulate barstate.isconfirmed with shift)
    df['is_confirmed'] = True  # Assume all bars are confirmed for simplicity
    
    # Map 4H conditions back to main timeframe
    df['bfvg_4h'] = bfvg_condition.reindex(df['datetime_index']).ffill().fillna(False)
    df['sfvg_4h'] = sfvg_condition.reindex(df['datetime_index']).ffill().fillna(False)
    
    # Iterate through bars to generate entries
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Update swing tracking
        if i > 0 and pd.notna(dailyHigh11.iloc[i]) and pd.notna(dailyHigh22.iloc[i]):
            if dailyHigh21.iloc[i] < dailyHigh22.iloc[i] and dailyHigh11.iloc[i+3] < dailyHigh22.iloc[i] and dailyHigh11.iloc[i+4] < dailyHigh22.iloc[i]:
                last_swing_high11 = dailyHigh22.iloc[i]
                last_swing_type = "dailyHigh"
        
        if i > 0 and pd.notna(dailyLow11.iloc[i]) and pd.notna(dailyLow22.iloc[i]):
            if dailyLow21.iloc[i] > dailyLow22.iloc[i] and dailyLow11.iloc[i+3] > dailyLow22.iloc[i] and dailyLow11.iloc[i+4] > dailyLow22.iloc[i]:
                last_swing_low11 = dailyLow22.iloc[i]
                last_swing_type = "dailyLow"
        
        # Check entry conditions
        if row['is_confirmed'] and in_trading_window.iloc[i]:
            entry_direction = None
            
            # Bullish FVG entry: BFVG condition + after daily low swing
            if row['bfvg_4h'] and last_swing_type == "dailyLow":
                entry_direction = "long"
            
            # Bearish FVG entry: SFVG condition + after daily high swing
            if row['sfvg_4h'] and last_swing_type == "dailyHigh":
                entry_direction = "short"
            
            if entry_direction:
                entry_price = row['close']
                entry_ts = int(row['time'])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                
                entries.append({
                    'trade_num': trade_num,
                    'direction': entry_direction,
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