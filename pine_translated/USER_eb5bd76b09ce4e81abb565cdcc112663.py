import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.
    """
    entries = []
    trade_num = 1
    
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Europe/London time windows converted to UTC
    # Window 1: 07:45-09:45 UTC
    # Window 2: 14:45-16:45 UTC
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['total_minutes'] = df['hour'] * 60 + df['minute']
    
    window1_start = 7 * 60 + 45
    window1_end = 9 * 60 + 45
    window2_start = 14 * 60 + 45
    window2_end = 16 * 60 + 45
    
    in_window = (
        ((df['total_minutes'] >= window1_start) & (df['total_minutes'] < window1_end)) |
        ((df['total_minutes'] >= window2_start) & (df['total_minutes'] < window2_end))
    )
    
    # Resample to 240-minute bars for security function simulation
    df_240 = df.set_index('datetime').resample('240T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'time': 'first'
    }).dropna(subset=['high', 'low', 'close']).reset_index()
    
    # 240-minute security data
    df_240['dailyHigh'] = df_240['high']
    df_240['dailyLow'] = df_240['low']
    df_240['dailyClose'] = df_240['close']
    df_240['dailyOpen'] = df_240['open']
    
    # Previous bar data (high[1], low[1])
    df_240['dailyHigh1'] = df_240['dailyHigh'].shift(1)
    df_240['dailyLow1'] = df_240['dailyLow'].shift(1)
    
    # Two bars back (high[2], low[2])
    df_240['dailyHigh2'] = df_240['dailyHigh'].shift(2)
    df_240['dailyLow2'] = df_240['dailyLow'].shift(2)
    
    # swing_detection function: is_swing_high = dailyHigh1 < dailyHigh2 and dailyHigh[3] < dailyHigh2 and dailyHigh[4] < dailyHigh2
    df_240['is_swing_high'] = (
        (df_240['dailyHigh1'] < df_240['dailyHigh2']) & 
        (df_240['dailyHigh'].shift(3) < df_240['dailyHigh2']) & 
        (df_240['dailyHigh'].shift(4) < df_240['dailyHigh2'])
    )
    
    # is_swing_low = dailyLow1 > dailyLow2 and dailyLow[3] > dailyLow2 and dailyLow[4] > dailyLow2
    df_240['is_swing_low'] = (
        (df_240['dailyLow1'] > df_240['dailyLow2']) & 
        (df_240['dailyLow'].shift(3) > df_240['dailyLow2']) & 
        (df_240['dailyLow'].shift(4) > df_240['dailyLow2'])
    )
    
    # Track last swing type
    df_240['lastSwingType1'] = 'none'
    prev_swing = 'none'
    for i in range(len(df_240)):
        if df_240['is_swing_high'].iloc[i]:
            prev_swing = 'dailyHigh'
        elif df_240['is_swing_low'].iloc[i]:
            prev_swing = 'dailyLow'
        df_240.iloc[i, df_240.columns.get_loc('lastSwingType1')] = prev_swing
    
    # FVG conditions: bfvg = dailyLow > dailyHigh2, sfvg = dailyHigh < dailyLow2
    df_240['bfvg'] = df_240['dailyLow'] > df_240['dailyHigh2']
    df_240['sfvg'] = df_240['dailyHigh'] < df_240['dailyLow2']
    
    # Entry conditions: bfvg + in_window + lastSwingType1 == "dailyLow", sfvg + in_window + lastSwingType1 == "dailyHigh"
    df_240['bullish_entry'] = df_240['bfvg'] & in_window.reindex(df_240.index, fill_value=False) & (df_240['lastSwingType1'] == 'dailyLow')
    df_240['bearish_entry'] = df_240['sfvg'] & in_window.reindex(df_240.index, fill_value=False) & (df_240['lastSwingType1'] == 'dailyHigh')
    
    for i in range(len(df_240)):
        if df_240['bullish_entry'].iloc[i] or df_240['bearish_entry'].iloc[i]:
            ts = int(df_240['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = df_240['close'].iloc[i]
            
            direction = 'long' if df_240['bullish_entry'].iloc[i] else 'short'
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
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