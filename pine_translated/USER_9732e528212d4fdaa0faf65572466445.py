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
    # Initialize variables
    entries = []
    trade_num = 1
    
    # Create 240 timeframe data using resample
    # First, identify FVG zones on 240tf
    # Bullish FVG: low > high[2] (current low is above candle before prev)
    # Bearish FVG: high < low[2]
    
    # Resample to 240 minutes for FVG detection
    df['time_dt'] = pd.to_datetime(df['time'], unit='ms')
    df_240 = df.set_index('time_dt').resample('240T').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    df_240['time'] = df_240.index.astype(np.int64) // 10**6
    
    # Calculate FVGs on 240 timeframe
    # Bullish FVG: low > high[2]
    # Bearish FVG: high < low[2]
    
    df_240['bull_fvg'] = df_240['low'] > df_240['high'].shift(2)
    df_240['bear_fvg'] = df_240['high'] < df_240['low'].shift(2)
    
    # Track FVG boxes
    bull_boxes = []
    bear_boxes = []
    
    for i in range(2, len(df_240)):
        if df_240['bull_fvg'].iloc[i]:
            fvg_top = df_240['low'].iloc[i]
            fvg_bottom = df_240['high'].iloc[i-2]
            bull_boxes.append({'top': fvg_top, 'bottom': fvg_bottom, 'time': df_240['time'].iloc[i]})
        
        if df_240['bear_fvg'].iloc[i]:
            fvg_top = df_240['high'].iloc[i]
            fvg_bottom = df_240['low'].iloc[i-2]
            bear_boxes.append({'top': fvg_top, 'bottom': fvg_bottom, 'time': df_240['time'].iloc[i]})
    
    # Check for entry signals on original timeframe
    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        
        # Check bull boxes for entry
        for box in bull_boxes:
            if box['time'] <= df['time'].iloc[i]:
                if df['low'].iloc[i] < box['top'] and df['low'].iloc[i] > box['bottom']:
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': df['time'].iloc[i],
                        'entry_time': datetime.fromtimestamp(df['time'].iloc[i]/1000, tz=timezone.utc).isoformat(),
                        'entry_price_guess': current_price,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': current_price,
                        'raw_price_b': current_price
                    })
                    trade_num += 1
        
        # Check bear boxes for entry
        for box in bear_boxes:
            if box['time'] <= df['time'].iloc[i]:
                if df['high'].iloc[i] > box['bottom'] and df['high'].iloc[i] < box['top']:
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': df['time'].iloc[i],
                        'entry_time': datetime.fromtimestamp(df['time'].iloc[i]/1000, tz=timezone.utc).isoformat(),
                        'entry_price_guess': current_price,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': current_price,
                        'raw_price_b': current_price
                    })
                    trade_num += 1
    
    return entries