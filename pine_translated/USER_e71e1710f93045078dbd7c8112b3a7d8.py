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
    
    # Helper functions
    def is_up(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]
    
    def is_down(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]
    
    def is_ob_up(idx):
        return is_down(idx + 1) and is_up(idx) and df['close'].iloc[idx] > df['high'].iloc[idx + 1]
    
    def is_ob_down(idx):
        return is_up(idx + 1) and is_down(idx) and df['close'].iloc[idx] < df['low'].iloc[idx + 1]
    
    def is_fvg_up(idx):
        return df['low'].iloc[idx] > df['high'].iloc[idx + 2]
    
    def is_fvg_down(idx):
        return df['high'].iloc[idx] < df['low'].iloc[idx + 2]
    
    # Convert timestamp to time for filtering
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    
    # Time windows: 07:00-09:59 and 12:00-14:59
    in_session1 = (df['time_minutes'] >= 7 * 60) & (df['time_minutes'] <= 9 * 60 + 59)
    in_session2 = (df['time_minutes'] >= 12 * 60) & (df['time_minutes'] <= 14 * 60 + 59)
    in_session = in_session1 | in_session2
    
    # Detect new day
    df['day'] = df['datetime'].dt.date
    is_new_day = df['day'] != df['day'].shift(1)
    is_new_day.iloc[0] = True
    
    # Calculate previous day high and low using shift
    df['prev_day_high'] = df['high'].shift(1).where(is_new_day.shift(1).fillna(False))
    df['prev_day_low'] = df['low'].shift(1).where(is_new_day.shift(1).fillna(False))
    
    # Carry forward prev day high/low within same day
    df['prev_day_high'] = df.groupby('day')['prev_day_high'].transform('first')
    df['prev_day_low'] = df.groupby('day')['prev_day_low'].transform('first')
    
    # Fill initial NaN values
    df['prev_day_high'] = df['prev_day_high'].ffill()
    df['prev_day_low'] = df['prev_day_low'].ffill()
    
    # Flags for liquidity sweeps - reset on new day
    flag_pdh = False
    flag_pdl = False
    
    # Calculate OB and FVG conditions
    ob_up = pd.Series(False, index=df.index)
    ob_down = pd.Series(False, index=df.index)
    fvg_up = pd.Series(False, index=df.index)
    fvg_down = pd.Series(False, index=df.index)
    
    for i in range(3, len(df)):
        try:
            ob_up.iloc[i] = is_ob_up(i)
            ob_down.iloc[i] = is_ob_down(i)
            fvg_up.iloc[i] = is_fvg_up(i)
            fvg_down.iloc[i] = is_fvg_down(i)
        except:
            pass
    
    # Stacked OB+FVG conditions
    stacked_bullish = ob_up & fvg_up
    stacked_bearish = ob_down & fvg_down
    
    # Iterate through bars to detect sweeps and generate entries
    for i in range(2, len(df)):
        current_ts = int(df['time'].iloc[i])
        current_close = df['close'].iloc[i]
        
        # Check for new day - reset flags
        if is_new_day.iloc[i]:
            flag_pdh = False
            flag_pdl = False
        
        # Detect previous day high sweep
        if current_close > df['prev_day_high'].iloc[i]:
            flag_pdh = True
        
        # Detect previous day low sweep
        if current_close < df['prev_day_low'].iloc[i]:
            flag_pdl = True
        
        # Check if we're in valid time session
        if not in_session.iloc[i]:
            continue
        
        # Skip if indicators are NaN
        if pd.isna(df['prev_day_high'].iloc[i]) or pd.isna(df['prev_day_low'].iloc[i]):
            continue
        
        # Long entry: bullish stacked OB+FVG after PDL sweep
        if stacked_bullish.iloc[i] and flag_pdl:
            entry_price = current_close
            entry_time = datetime.fromtimestamp(current_ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': current_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            flag_pdl = False  # Reset after entry
        
        # Short entry: bearish stacked OB+FVG after PDH sweep
        if stacked_bearish.iloc[i] and flag_pdh:
            entry_price = current_close
            entry_time = datetime.fromtimestamp(current_ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': current_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            flag_pdh = False  # Reset after entry
    
    return entries