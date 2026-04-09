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
    
    # Reset index to ensure 0-based integer indexing
    df = df.reset_index(drop=True)
    
    # --- Compute previous day data for BIAS calculation ---
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['dt'].dt.date
    
    # Get daily OHLC
    daily = df.groupby('date').agg(
        day_open=('open', 'first'),
        day_high=('high', 'max'),
        day_low=('low', 'min'),
        day_close=('close', 'last')
    )
    
    # Create shifted prev day columns
    daily['prev_high'] = daily['day_high'].shift(1)
    daily['prev_low'] = daily['day_low'].shift(1)
    daily['prev_close'] = daily['day_close'].shift(1)
    
    # Merge back to main df
    df = df.merge(daily[['prev_high', 'prev_low', 'prev_close', 'day_open']], left_on='date', right_index=True, how='left')
    
    # --- Calculate BIAS ---
    day_open = df['day_open']
    prev_close = df['prev_close']
    midpoint = (df['prev_high'] + df['prev_low']) / 2.0
    
    bias_up = (day_open > prev_close) & (day_open > midpoint)
    bias_down = (day_open < prev_close) & (day_open < midpoint)
    bias = pd.Series(np.where(bias_up, 'Bullish', np.where(bias_down, 'Bearish', 'Neutral')), index=df.index)
    
    # --- Detect FVGs ---
    bullish_fvg = (df['high'].shift(2) < df['low'])
    bearish_fvg = (df['low'].shift(2) > df['high'])
    
    # --- State variables ---
    ig_active = False
    ig_direction = 0
    ig_c1_high = np.nan
    ig_c1_low = np.nan
    ig_validation_end = np.nan
    
    trade_num = 1
    entries = []
    
    # --- Iterate bars (need 2 bars back for FVG) ---
    for i in range(2, len(df)):
        current_ts = int(df['time'].iloc[i])
        current_bias = bias.iloc[i]
        current_close = df['close'].iloc[i]
        
        # Skip if any required values are NaN
        if pd.isna(df['high'].iloc[i-2]) or pd.isna(df['low'].iloc[i-2]):
            continue
        if pd.isna(prev_close.iloc[i]) or pd.isna(midpoint.iloc[i]):
            continue
        
        # Detect bullish FVG
        if bullish_fvg.iloc[i]:
            ig_active = True
            ig_direction = -1
            ig_c1_high = df['high'].iloc[i-2]
            ig_c1_low = df['low'].iloc[i-2]
            ig_validation_end = i + 4
        
        # Detect bearish FVG
        elif bearish_fvg.iloc[i]:
            ig_active = True
            ig_direction = 1
            ig_c1_high = df['high'].iloc[i-2]
            ig_c1_low = df['low'].iloc[i-2]
            ig_validation_end = i + 4
        
        # Validate FVG within window
        validated = False
        if ig_active and i <= ig_validation_end:
            if ig_direction == 1 and current_close < ig_c1_high:
                validated = True
            if ig_direction == -1 and current_close > ig_c1_low:
                validated = True
        
        # Entry conditions
        if validated:
            if ig_direction == -1 and current_bias == 'Bullish':
                entry_price = current_close
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': current_ts,
                    'entry_time': datetime.fromtimestamp(current_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
            
            elif ig_direction == 1 and current_bias == 'Bearish':
                entry_price = current_close
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': current_ts,
                    'entry_time': datetime.fromtimestamp(current_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
            
            ig_active = False
    
    return entries