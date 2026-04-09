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
    results = []
    trade_num = 1
    
    # Calculate ATR (Wilder)
    tr = np.maximum(df['high'] - df['low'], 
                    np.maximum(np.abs(df['high'] - df['close'].shift(1)),
                               np.abs(df['low'] - df['close'].shift(1)))
                   )
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # Previous day high/low (shifted by 1 day approximated by checking day change)
    df['day'] = pd.to_datetime(df['time'], unit='s').dt.date
    df['prev_day_high'] = df['high'].shift(1).where(df['day'].shift(1) != df['day'])
    df['prev_day_high'] = df['prev_day_high'].ffill()
    df['prev_day_low'] = df['low'].shift(1).where(df['day'].shift(1) != df['day'])
    df['prev_day_low'] = df['prev_day_low'].ffill()
    
    # London trading windows (7:45-9:45 and 15:45-16:45 UTC)
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    
    morning_window = ((df['hour'] == 7) & (df['minute'] >= 45)) | \
                     ((df['hour'] == 8)) | \
                     ((df['hour'] == 9) & (df['minute'] <= 45))
    
    afternoon_window = ((df['hour'] == 15) & (df['minute'] >= 45)) | \
                       ((df['hour'] == 16) & (df['minute'] <= 45))
    
    in_trading_window = morning_window | afternoon_window
    
    # Liquidity sweep conditions
    # Short entry: price sweeps above previous day high (bearish liquidity claim)
    price_sweeps_prev_high = df['high'] > df['prev_day_high']
    
    # Long entry: price sweeps below previous day low (bullish liquidity claim)
    price_sweeps_prev_low = df['low'] < df['prev_day_low']
    
    # Entry conditions
    long_condition = price_sweeps_prev_low & in_trading_window & df['prev_day_low'].notna()
    short_condition = price_sweeps_prev_high & in_trading_window & df['prev_day_high'].notna()
    
    # Track if already in position to avoid duplicate entries
    in_long = False
    in_short = False
    
    for i in range(len(df)):
        if pd.isna(df['high'].iloc[i]) or pd.isna(df['low'].iloc[i]):
            continue
            
        # Check long entry
        if long_condition.iloc[i] and not in_long:
            entry_price = df['close'].iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
            in_long = True
            in_short = False
        
        # Check short entry
        elif short_condition.iloc[i] and not in_short:
            entry_price = df['close'].iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
            in_short = True
            in_long = False
        
        # Reset position tracking if close crosses back (simple reset)
        if in_long and df['close'].iloc[i] > df['prev_day_high'].iloc[i]:
            in_long = False
        if in_short and df['close'].iloc[i] < df['prev_day_low'].iloc[i]:
            in_short = False
    
    return results