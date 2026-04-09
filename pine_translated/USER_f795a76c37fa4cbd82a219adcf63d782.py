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
    
    # Create shifted Series for FVG detection
    # Pine Script: [0]=current, [1]=previous, [2]=2 bars back
    close_0 = df['close']
    close_1 = df['close'].shift(1)
    close_2 = df['close'].shift(2)
    
    open_1 = df['open'].shift(1)
    
    high_0 = df['high']
    high_1 = df['high'].shift(1)
    high_2 = df['high'].shift(2)
    
    low_0 = df['low']
    low_1 = df['low'].shift(1)
    low_2 = df['low'].shift(2)
    
    # Default inputs from Pine Script
    FVGBKWY_on = True
    FVGImb = True
    enter_trades = True
    
    # Bearish FVG (Top Imbalance Breakaway)
    # TopImbalance_Bway: low[2] <= open[1] AND high[0] >= close[1] AND close[0] < low[1]
    top_imbalance_bway = (
        (low_2 <= open_1) & 
        (high_0 >= close_1) & 
        (close_0 < low_1)
    )
    top_imbalance_size = low_2 - high_0
    
    # Bullish FVG (Bottom Imbalance Breakaway)
    # BottomInbalance_Bway: high[2] >= open[1] AND low[0] <= close[1] AND close[0] > high[1]
    bottom_imbalance_bway = (
        (high_2 >= open_1) & 
        (low_0 <= close_1) & 
        (close_0 > high_1)
    )
    bottom_imbalance_size = low_0 - high_2
    
    # Build entry conditions
    short_condition = (
        FVGBKWY_on & 
        FVGImb & 
        top_imbalance_bway & 
        (top_imbalance_size > 0) &
        enter_trades
    )
    
    long_condition = (
        FVGBKWY_on & 
        FVGImb & 
        bottom_imbalance_bway & 
        (bottom_imbalance_size > 0) &
        enter_trades
    )
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(close_0.iloc[i]):
            continue
        
        if short_condition.iloc[i] if not pd.isna(short_condition.iloc[i]) else False:
            ts = int(df['time'].iloc[i])
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif long_condition.iloc[i] if not pd.isna(long_condition.iloc[i]) else False:
            ts = int(df['time'].iloc[i])
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries