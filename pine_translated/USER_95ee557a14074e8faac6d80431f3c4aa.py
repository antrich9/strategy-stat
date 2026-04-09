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
    
    # Input parameters from Pine Script
    FVGBKWY_on = True
    FVGImb = True
    FVGnew = True
    
    # Calculate required shift values for FVG detection
    # Pine Script uses negative indexing: low[2] = low shifted back 2 bars
    # In pandas, this means shift(2) for 2 bars back
    
    open_1 = df['open'].shift(1)
    close_1 = df['close'].shift(1)
    low_1 = df['low'].shift(1)
    high_1 = df['high'].shift(1)
    low_2 = df['low'].shift(2)
    high_2 = df['high'].shift(2)
    
    # Top Imbalance (Bearish FVG): gap down from previous candle
    # TopImbalance_Bway = low[2] <= open[1] and high[0] >= close[1] and close[0] < low[1]
    top_imbalance_bway = (low_2 <= open_1) & (df['high'] >= close_1) & (df['close'] < low_1)
    
    # Top_ImbXBway = low[2] <= open[1] and high[0] >= close[1] and close[0] > low[1]
    top_imb_xbway = (low_2 <= open_1) & (df['high'] >= close_1) & (df['close'] > low_1)
    
    # TopImbalancesize = low[2] - high[0]
    top_imbalance_size = low_2 - df['high']
    
    # Bottom Inbalance (Bullish FVG): gap up from previous candle
    # BottomInbalance_Bway = high[2] >= open[1] and low[0] <= close[1] and close[0] > high[1]
    bottom_inbalance_bway = (high_2 >= open_1) & (df['low'] <= close_1) & (df['close'] > high_1)
    
    # Bottom_ImbXBAway = high[2] >= open[1] and low[0] <= close[1] and close[0] < high[1]
    bottom_imb_xbway = (high_2 >= open_1) & (df['low'] <= close_1) & (df['close'] < high_1)
    
    # BottomInbalancesize = low[0] - high[2]
    bottom_inbalance_size = df['low'] - high_2
    
    # Short entry conditions (bearish FVG patterns)
    short_condition = (
        FVGBKWY_on & FVGImb & top_imbalance_bway & (top_imbalance_size > 0) |
        FVGBKWY_on & FVGnew & top_imb_xbway & (top_imbalance_size > 0)
    )
    
    # Long entry conditions (bullish FVG patterns)
    long_condition = (
        FVGBKWY_on & FVGImb & bottom_inbalance_bway & (bottom_inbalance_size > 0) |
        FVGBKWY_on & FVGnew & bottom_imb_xbway & (bottom_inbalance_size > 0)
    )
    
    entries = []
    trade_num = 1
    
    # Iterate through bars starting from index 2 (to have data for shift(2))
    for i in range(2, len(df)):
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
        entry_price = df['close'].iloc[i]
        
        # Check short entry
        if short_condition.iloc[i]:
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
        
        # Check long entry
        if long_condition.iloc[i]:
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
    
    return entries