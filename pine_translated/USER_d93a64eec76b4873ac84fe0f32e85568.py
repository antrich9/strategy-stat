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
    # Strategy parameters from Pine Script
    fastLength = 50
    slowLength = 200
    start_hour = 7
    end_hour = 10
    end_minute = 59
    
    # Create copies to avoid modifying original
    data = df.copy()
    
    # Calculate EMAs
    ema_fast = data['close'].ewm(span=fastLength, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slowLength, adjust=False).mean()
    
    # Calculate previous day high and low using resampling
    data['datetime'] = pd.to_datetime(data['time'], unit='ms', utc=True)
    data['day'] = data['datetime'].dt.date
    
    # Get previous day's high and low
    prev_day_hl = data.groupby('day')['high'].shift(1)
    prev_day_ll = data.groupby('day')['low'].shift(1)
    
    prevDayHigh = data.groupby('day')['high'].transform(lambda x: x.shift(1).iloc[0] if len(x) > 0 else np.nan)
    prevDayLow = data.groupby('day')['low'].transform(lambda x: x.shift(1).iloc[0] if len(x) > 0 else np.nan)
    
    # Recalculate properly
    daily_high = data.groupby('day')['high'].first().shift(1)
    daily_low = data.groupby('day')['low'].first().shift(1)
    
    data['prevDayHigh'] = data['day'].map(daily_high)
    data['prevDayLow'] = data['day'].map(daily_low)
    
    # Extract hour and minute from timestamp
    data['hour'] = data['datetime'].dt.hour
    data['minute'] = data['datetime'].dt.minute
    data['month'] = data['datetime'].dt.month
    data['dayofweek'] = data['datetime'].dt.dayofweek
    data['dayofmonth'] = data['datetime'].dt.day
    
    # DST check (simplified for UTC+1)
    is_dst = ((data['month'] >= 3) & (data['month'] == 3) & (data['dayofweek'] == 6) & (data['dayofmonth'] >= 25) | (data['month'] > 3)) & \
             ((data['month'] <= 10) & (data['month'] == 10) & (data['dayofweek'] == 6) & (data['dayofmonth'] < 25) | (data['month'] < 10))
    
    # Adjust hour for UTC+1 and DST
    adjusted_hour = data['hour'] + 1 + is_dst.astype(int)
    adjusted_minute = data['minute']
    
    # Trading window check
    in_trading_window = (adjusted_hour >= start_hour) & (adjusted_hour <= end_hour) & \
                        ~((adjusted_hour == end_hour) & (adjusted_minute > end_minute))
    
    # Detect new day
    data['isNewDay'] = data['day'] != data['day'].shift(1)
    
    # Initialize flags
    flagpdl = pd.Series(False, index=data.index)  # Previous day low swept
    flagpdh = pd.Series(False, index=data.index)  # Previous day high swept
    
    # Process sweeps and flags
    for i in range(1, len(data)):
        if data['isNewDay'].iloc[i]:
            flagpdl.iloc[i] = False
            flagpdh.iloc[i] = False
        else:
            flagpdl.iloc[i] = flagpdl.iloc[i-1]
            flagpdh.iloc[i] = flagpdh.iloc[i-1]
        
        if pd.notna(data['prevDayHigh'].iloc[i]):
            if data['close'].iloc[i] > data['prevDayHigh'].iloc[i]:
                flagpdh.iloc[i] = True
            if data['close'].iloc[i] < data['prevDayLow'].iloc[i]:
                flagpdl.iloc[i] = True
    
    # Entry conditions based on ICT concepts:
    # Long: price sweeps previous day low (flagpdl=True) during trading window, EMA crossover (fast > slow)
    # Short: price sweeps previous day high (flagpdh=True) during trading window, EMA crossunder (fast < slow)
    
    # EMA crossover for long
    ema_fast_above_slow = ema_fast > ema_slow
    ema_fast_below_slow = ema_fast < ema_slow
    
    # EMA crossover condition
    ema_crossup = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    ema_crossdown = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
    
    # Long entry: flagpdl (low swept), in trading window, EMA trending up (fast > slow)
    long_condition = flagpdl & in_trading_window & ema_fast_above_slow
    
    # Short entry: flagpdh (high swept), in trading window, EMA trending down (fast < slow)
    short_condition = flagpdh & in_trading_window & ema_fast_below_slow
    
    # Build entries list
    entries = []
    trade_num = 1
    
    for i in range(1, len(data)):
        if i < slowLength:
            continue
            
        entry_price = data['close'].iloc[i]
        ts = int(data['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
        
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
            
        elif short_condition.iloc[i]:
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
    
    return entries