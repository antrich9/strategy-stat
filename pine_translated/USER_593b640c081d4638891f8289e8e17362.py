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
    df = df.copy()
    df.reset_index(drop=True, inplace=True)
    
    # Extract hour and minute for kill zone check
    # kill_zone_start = 09:00 (540 minutes), kill_zone_end = 17:00 (1020 minutes)
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['kill_zone'] = ((df['hour'] * 60 + df['minute']) >= 540) & ((df['hour'] * 60 + df['minute']) <= 1020)
    
    # Calculate moving averages
    df['trend_ma'] = df['close'].rolling(50).mean()
    df['pullback_ma'] = df['close'].rolling(14).mean()
    
    entries = []
    trade_num = 0
    
    for i in range(1, len(df)):
        # Skip if indicators are NaN
        if pd.isna(df['trend_ma'].iloc[i]) or pd.isna(df['pullback_ma'].iloc[i]):
            continue
        
        kill_zone = df['kill_zone'].iloc[i]
        
        # Get current and previous values
        close_curr = df['close'].iloc[i]
        close_prev = df['close'].iloc[i-1]
        pullback_ma_curr = df['pullback_ma'].iloc[i]
        pullback_ma_prev = df['pullback_ma'].iloc[i-1]
        
        # Long condition: up trend + kill zone + crossover(close, pullback_ma)
        crossover = (close_prev <= pullback_ma_prev) and (close_curr > pullback_ma_curr)
        if kill_zone and crossover:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close_curr,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_curr,
                'raw_price_b': close_curr
            })
        
        # Short condition: down trend + kill zone + crossunder(close, pullback_ma)
        crossunder = (close_prev >= pullback_ma_prev) and (close_curr < pullback_ma_curr)
        if kill_zone and crossunder:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close_curr,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_curr,
                'raw_price_b': close_curr
            })
    
    return entries