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
    
    # Create datetime column for time filtering
    df = df.copy()
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Time window: London sessions (Morning 08:00-09:55, Afternoon 14:00-16:55)
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    is_within_morning = (df['hour'] == 8) & (df['minute'] < 55)
    is_within_afternoon = (df['hour'] >= 14) & (df['hour'] < 17) & ((df['hour'] == 14) | (df['hour'] < 16) | ((df['hour'] == 16) & (df['minute'] < 55)))
    in_trading_window = is_within_morning | is_within_afternoon
    
    # FVG detection on 15m TF
    high_arr = df['high'].values
    low_arr = df['low'].values
    x_fvg = np.where(low_arr[2:] >= high_arr[:-2], 1, 
                     np.where(low_arr[:-2] >= high_arr[2:], -1, 0))
    x_fvg = np.concatenate([[0, 0], x_fvg])
    is_bullish_fvg = x_fvg > 0
    is_bearish_fvg = x_fvg < 0
    
    # Swing detection (local high/low with 4-bar lookback)
    is_swing_high = np.zeros(len(df), dtype=bool)
    is_swing_low = np.zeros(len(df), dtype=bool)
    for i in range(4, len(df)):
        main_high = high_arr[i-2]
        main_low = low_arr[i-2]
        if high_arr[i-1] < main_high and high_arr[i-3] < main_high and high_arr[i-4] < main_high:
            is_swing_high[i] = True
        if low_arr[i-1] > main_low and low_arr[i-3] > main_low and low_arr[i-4] > main_low:
            is_swing_low[i] = True
    
    # EMA indicators (simulate request.security for 5 TF by using 10/50 ema on available data)
    ema_fast = df['close'].ewm(span=10, adjust=False).mean()
    ema_slow = df['close'].ewm(span=50, adjust=False).mean()
    ema_bullish = ema_fast > ema_slow
    ema_bearish = ema_fast < ema_slow
    
    # Build conditions
    long_condition = ema_bullish & is_bullish_fvg & is_swing_low & in_trading_window
    short_condition = ema_bearish & is_bearish_fvg & is_swing_high & in_trading_window
    
    for i in range(5, len(df)):
        if pd.isna(ema_fast.iloc[i]) or pd.isna(ema_slow.iloc[i]):
            continue
        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries