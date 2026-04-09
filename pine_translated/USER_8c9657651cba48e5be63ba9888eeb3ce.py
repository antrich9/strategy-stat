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
    
    # Deep copy to avoid modifying original df
    data = df.copy()
    
    # 1. Calculate Previous Day High/Low
    # Assuming df['time'] is unix timestamp (UTC)
    data['dt'] = pd.to_datetime(data['time'], unit='s')
    data['date'] = data['dt'].dt.date
    
    # Aggregate daily high and low
    daily_agg = data.groupby('date').agg({'high': 'max', 'low': 'min'})
    daily_agg = daily_agg.shift(1) # Shift to get previous day values
    daily_agg.columns = ['prev_day_high', 'prev_day_low']
    
    # Merge back
    data = data.merge(daily_agg, on='date', how='left')
    
    # 2. Calculate Wilder ATR (len=144)
    high = data['high']
    low = data['low']
    close = data['close']
    
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1.0/144, adjust=False).mean()
    
    # Input filter value
    fvgTH = 0.5
    filtered_atr = atr * fvgTH
    
    # 3. Calculate Conditions
    
    # bullG = low > high.shift(1)
    bullG = low > high.shift(1)
    # bearG = high < low.shift(1)
    bearG = high < low.shift(1)
    
    # Bull Condition Components
    # (b.l - b.h[2]) > atr
    comp1_bull = (low - high.shift(2)) > filtered_atr
    # b.l > b.h[2]
    comp2_bull = low > high.shift(2)
    # b.c[1] > b.h[2]
    comp3_bull = close.shift(1) > high.shift(2)
    # not (bullG or bullG[1])
    comp4_bull = ~(bullG | bullG.shift(1))
    
    bull = comp1_bull & comp2_bull & comp3_bull & comp4_bull
    
    # Bear Condition Components
    # (