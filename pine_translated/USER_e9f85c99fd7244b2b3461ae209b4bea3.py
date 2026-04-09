import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    if len(df) < 3:
        return []
    
    df = df.copy().reset_index(drop=True)
    
    # Calculate EMAs
    df['fastEMA'] = df['close'].ewm(span=50, adjust=False).mean()
    df['slowEMA'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # Detect EMA crossovers
    df['crossover'] = (df['fastEMA'] > df['slowEMA']) & (df['fastEMA'].shift(1) <= df['slowEMA'].shift(1))
    df['crossunder'] = (df['fastEMA'] < df['slowEMA']) & (df['fastEMA'].shift(1) >= df['slowEMA'].shift(1))
    
    # Candlestick patterns
    is_up = (df['close'] > df['open'])
    is_down = (df['close'] < df['open'])
    
    # Bullish OB at bar 1: bar 2 bearish, bar 1 bullish, close[1] > high[2]
    ob_up = is_down.shift(2) & is_up.shift(1) & (df['close'].shift(1) > df['high'].shift(2))
    
    # Bearish OB at bar 1: bar 2 bullish, bar 1 bearish, close[1] < low[2]
    ob_down = is_up.shift(2) & is_down.shift(1) & (df['close'].shift(1) < df['low'].shift(2))
    
    # Bullish FVG at bar 0: low[0] > high[2]
    fvg_up = (df['low'] > df['high'].shift(2))
    
    # Bearish FVG at bar 0: high[0] < low[2]
    fvg_down = (df['high'] < df['low'].shift(2))
    
    # Combined patterns
    bullish_pattern = ob_up & fvg_up
    bearish_pattern = ob_down & fvg_down
    
    entries = []
    trade_num = 1
    
    waiting_for_long = False
    waiting_for_short = False
    
    for i in range(3, len(df)):
        ts = int(df['time'].iloc[i])
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        
        # Time filter
        in_time_window = (7 <= hour < 10) or (12 <= hour < 15)
        
        # Check for bullish entry conditions
        if df['crossover'].iloc[i] and in_time_window and bullish_pattern.iloc[i]:
            waiting_for_long = True
            entry_price = float(df['close'].iloc[i])
        
        # Enter long if waiting and pattern confirms
        if waiting_for_long and bullish_pattern.iloc[i] and in_time_window:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': dt.isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            waiting_for_long = False
        
        # Check for bearish entry conditions
        if df['crossunder'].iloc[i] and in_time_window and bearish_pattern.iloc[i]:
            waiting_for_short = True
            entry_price = float(df['close'].iloc[i])
        
        # Enter short if waiting and pattern confirms
        if waiting_for_short and bearish_pattern.iloc[i] and in_time_window:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': dt.isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            waiting_for_short = False
    
    return entries