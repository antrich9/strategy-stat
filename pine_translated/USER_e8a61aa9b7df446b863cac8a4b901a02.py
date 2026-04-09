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
    
    # Ensure we have required columns
    required_cols = ['time', 'open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Need: {required_cols}")
    
    df = df.copy()
    n = len(df)
    
    # Convert time to datetime for timezone-aware operations
    df['datetime'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert('Europe/London')
    
    # Extract hour and minute in London timezone
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    
    # Define trading windows (London time)
    # Morning: 7:45 to 9:45
    # Afternoon: 15:45 to 16:45
    
    def in_trading_window(row):
        hour = row['hour']
        minute = row['minute']
        total_mins = hour * 60 + minute
        
        # Morning window: 7:45 (465) to 9:45 (585)
        morning_start = 7 * 60 + 45  # 465
        morning_end = 9 * 60 + 45    # 585
        
        # Afternoon window: 15:45 (945) to 16:45 (1005)
        afternoon_start = 15 * 60 + 45  # 945
        afternoon_end = 16 * 60 + 45    # 1005
        
        in_morning = morning_start <= total_mins < morning_end
        in_afternoon = afternoon_start <= total_mins < afternoon_end
        
        return in_morning or in_afternoon
    
    df['isWithinTimeWindow'] = df.apply(in_trading_window, axis=1)
    
    # Get previous day high and low using daily data
    df['prevDayHigh'] = df['high'].shift(1).rolling(window=2).max().shift(1)
    df['prevDayLow'] = df['low'].shift(1).rolling(window=2).min().shift(1)
    
    # Fill initial NaN values with forward/backward fill
    df['prevDayHigh'] = df['prevDayHigh'].fillna(method='ffill').fillna(method='bfill')
    df['prevDayLow'] = df['prevDayLow'].fillna(method='ffill').fillna(method='bfill')
    
    # Check for previous day high/low being taken
    df['previousDayHighTaken'] = df['high'] > df['prevDayHigh']
    df['previousDayLowTaken'] = df['low'] < df['prevDayLow']
    
    # Get current day high and low (using 240 min / 4H data approximation)
    df['currentDayHigh'] = df['high'].rolling(window=4).max()
    df['currentDayLow'] = df['low'].rolling(window=4).min()
    
    # Calculate flags (similar to var bool logic in Pine)
    df['flagpdh'] = False
    df['flagpdl'] = False
    
    # Iterate to handle the stateful nature of flags
    for i in range(1, n):
        if df['previousDayHighTaken'].iloc[i] and df['currentDayLow'].iloc[i] > df['prevDayLow'].iloc[i]:
            df.loc[df.index[i], 'flagpdh'] = True
        elif df['previousDayLowTaken'].iloc[i] and df['currentDayHigh'].iloc[i] < df['prevDayHigh'].iloc[i]:
            df.loc[df.index[i], 'flagpdl'] = True
        else:
            df.loc[df.index[i], 'flagpdh'] = False
            df.loc[df.index[i], 'flagpdl'] = False
    
    # Convert to boolean arrays for crossover calculations
    flagpdh = df['flagpdh']
    flagpdl = df['flagpdl']
    isWithinTimeWindow = df['isWithinTimeWindow']
    close = df['close']
    
    # Long entry condition: flagpdh is true and within time window
    # Short entry condition: flagpdl is true and within time window
    long_condition = flagpdh & isWithinTimeWindow
    short_condition = flagpdl & isWithinTimeWindow
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(n):
        # Skip if required indicators are NaN
        if pd.isna(df['prevDayHigh'].iloc[i]) or pd.isna(df['prevDayLow'].iloc[i]):
            continue
        if pd.isna(df['currentDayHigh'].iloc[i]) or pd.isna(df['currentDayLow'].iloc[i]):
            continue
        if pd.isna(close.iloc[i]):
            continue
        
        entry_ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
        entry_price = float(close.iloc[i])
        
        # Check for long entry
        if long_condition.iloc[i]:
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
        
        # Check for short entry
        if short_condition.iloc[i]:
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