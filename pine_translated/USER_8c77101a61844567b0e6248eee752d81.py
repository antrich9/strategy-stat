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
    
    # Extract hour and minute from timestamp for time window filtering
    ts_dt = pd.to_datetime(df['time'], unit='ms', utc=True)
    df['hour'] = ts_dt.dt.hour
    df['minute'] = ts_dt.dt.minute
    
    # London time window: 07:45-09:45 and 14:45-16:45
    morning_window = ((df['hour'] == 7) & (df['minute'] >= 45)) | \
                     ((df['hour'] >= 8) & (df['hour'] < 9)) | \
                     ((df['hour'] == 9) & (df['minute'] <= 45))
    
    afternoon_window = ((df['hour'] == 14) & (df['minute'] >= 45)) | \
                        ((df['hour'] >= 15) & (df['hour'] < 16)) | \
                        ((df['hour'] == 16) & (df['minute'] <= 45))
    
    in_trading_window = morning_window | afternoon_window
    
    # Bullish OB: isObUp(1) at current bar means checking bar i+1 and i+2
    # isDown(i+1) and isUp(i+2) and close[i+2] > high[i+1]
    is_down_iplus1 = df['close'].shift(1) < df['open'].shift(1)
    is_up_iplus2 = df['close'].shift(2) > df['open'].shift(2)
    close_above_prev_high = df['close'].shift(2) > df['high'].shift(1)
    ob_up = is_down_iplus1 & is_up_iplus2 & close_above_prev_high
    
    # Bearish OB: isObDown(1) at current bar means checking bar i+1 and i+2
    # isUp(i+1) and isDown(i+2) and close[i+2] < low[i+1]
    is_up_iplus1 = df['close'].shift(1) > df['open'].shift(1)
    is_down_iplus2 = df['close'].shift(2) < df['open'].shift(2)
    close_below_prev_low = df['close'].shift(2) < df['low'].shift(1)
    ob_down = is_up_iplus1 & is_down_iplus2 & close_below_prev_low
    
    # Bullish FVG: isFvgUp(0) checks low[0] > high[2]
    fvg_up = df['low'] > df['high'].shift(2)
    
    # Bearish FVG: isFvgDown(0) checks high[0] < low[2]
    fvg_down = df['high'] < df['low'].shift(2)
    
    # Stacked OB + FVG: previous bar has OB, current bar has FVG
    long_entry = ob_up.shift(1) & fvg_up & in_trading_window
    short_entry = ob_down.shift(1) & fvg_down & in_trading_window
    
    # Build entry list
    for idx in df[long_entry].index:
        ts = int(df.loc[idx, 'time'])
        entries.append({
            'trade_num': trade_num,
            'direction': 'long',
            'entry_ts': ts,
            'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
            'entry_price_guess': float(df.loc[idx, 'close']),
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': float(df.loc[idx, 'close']),
            'raw_price_b': float(df.loc[idx, 'close'])
        })
        trade_num += 1
    
    for idx in df[short_entry].index:
        ts = int(df.loc[idx, 'time'])
        entries.append({
            'trade_num': trade_num,
            'direction': 'short',
            'entry_ts': ts,
            'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
            'entry_price_guess': float(df.loc[idx, 'close']),
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': float(df.loc[idx, 'close']),
            'raw_price_b': float(df.loc[idx, 'close'])
        })
        trade_num += 1
    
    return entries