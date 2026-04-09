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
    
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.sort_values('time').reset_index(drop=True)
    
    # Simulate 4H data by resampling
    df_4h = df.set_index('time').resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    
    # Simulate Daily data by resampling
    df_daily = df.set_index('time').resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    
    # Get 4H data arrays
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    high_4h_1 = df_4h['high'].shift(1).values
    low_4h_1 = df_4h['low'].shift(1).values
    high_4h_2 = df_4h['high'].shift(2).values
    low_4h_2 = df_4h['low'].shift(2).values
    close_4h_1 = df_4h['close'].shift(1).values
    
    # Get Daily data arrays
    dailyHigh11 = df_daily['high'].values
    dailyLow11 = df_daily['low'].values
    dailyHigh21 = df_daily['high'].shift(1).values
    dailyLow21 = df_daily['low'].shift(1).values
    dailyHigh22 = df_daily['high'].shift(2).values
    dailyLow22 = df_daily['low'].shift(2).values
    
    # Swing detection: is_swing_high11 and is_swing_low11
    is_swing_high11 = (dailyHigh21 < dailyHigh22) & (np.roll(dailyHigh11, 3) < dailyHigh22) & (np.roll(dailyHigh11, 4) < dailyHigh22)
    is_swing_low11 = (dailyLow21 > dailyLow22) & (np.roll(dailyLow11, 3) > dailyLow22) & (np.roll(dailyLow11, 4) > dailyLow22)
    
    # Mark first few as False since rolling shifts create invalid values
    is_swing_high11[:5] = False
    is_swing_low11[:5] = False
    
    # Track last swing type
    lastSwingType11 = np.full(len(df_daily), "none", dtype=object)
    last_swing_high11 = np.full(len(df_daily), np.nan)
    last_swing_low11 = np.full(len(df_daily), np.nan)
    
    for i in range(len(df_daily)):
        if is_swing_high11[i]:
            last_swing_high11[i] = dailyHigh22[i]
            lastSwingType11[i] = "dailyHigh"
        elif is_swing_low11[i]:
            last_swing_low11[i] = dailyLow22[i]
            lastSwingType11[i] = "dailyLow"
        else:
            if i > 0:
                lastSwingType11[i] = lastSwingType11[i-1]
                last_swing_high11[i] = last_swing_high11[i-1]
                last_swing_low11[i] = last_swing_low11[i-1]
    
    # FVG conditions
    bfvg_condition = low_4h > high_4h_2
    sfvg_condition = high_4h < low_4h_2
    
    # Handle NaN at start of 4H series
    bfvg_condition[:2] = False
    sfvg_condition[:2] = False
    
    # Previous day high/low (pdHigh, pdLow) - shift daily data by 1
    pdHigh = np.roll(dailyHigh11, 1)
    pdLow = np.roll(dailyLow11, 1)
    pdHigh[0] = np.nan
    pdLow[0] = np.nan
    
    # Match daily swing type to 4H bars
    df_4h['swing_type'] = np.nan
    df_4h['swing_type'] = df_4h['swing_type'].astype(object)
    
    # Map daily swing type to 4H bars based on date alignment
    for i in range(len(df_4h)):
        bar_time = df_4h['time'].iloc[i]
        # Find matching daily bar
        for j in range(len(df_daily)):
            if df_daily['time'].iloc[j].date() == bar_time.date():
                df_4h.iloc[i, df_4h.columns.get_loc('swing_type')] = lastSwingType11[j]
                break
    
    # Create conditions for each 4H bar
    is_bullish_entry = bfvg_condition & (df_4h['swing_type'] == "dailyLow").values
    is_bearish_entry = sfvg_condition & (df_4h['swing_type'] == "dailyHigh").values
    
    # Also apply trading window filter (simulate London time windows)
    # Window 1: 07:45-11:45, Window 2: 14:00-14:45
    df_4h['hour'] = df_4h['time'].dt.hour
    df_4h['minute'] = df_4h['time'].dt.minute
    
    in_window1 = ((df_4h['hour'] == 7) & (df_4h['minute'] >= 45)) | ((df_4h['hour'] >= 8) & (df_4h['hour'] < 11)) | ((df_4h['hour'] == 11) & (df_4h['minute'] < 45))
    in_window2 = (df_4h['hour'] == 14) & (df_4h['minute'] <= 45)
    in_trading_window = in_window1 | in_window2
    
    # Apply window filter to entries
    is_bullish_entry = is_bullish_entry & in_trading_window.values
    is_bearish_entry = is_bearish_entry & in_trading_window.values
    
    entries = []
    trade_num = 1
    
    # Process bullish entries
    for i in range(len(df_4h)):
        if is_bullish_entry[i]:
            entry_ts = int(df_4h['time'].iloc[i].timestamp())
            entry_time = df_4h['time'].iloc[i].isoformat()
            entry_price = close_4h[i]
            
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
    
    # Process bearish entries
    for i in range(len(df_4h)):
        if is_bearish_entry[i]:
            entry_ts = int(df_4h['time'].iloc[i].timestamp())
            entry_time = df_4h['time'].iloc[i].isoformat()
            entry_price = close_4h[i]
            
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