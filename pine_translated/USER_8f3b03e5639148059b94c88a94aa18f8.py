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
    df['ts'] = df['time']
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    df['minute_of_day'] = df['hour'] * 60 + df['minute']
    
    london_morning_start = 8 * 60
    london_morning_end = 9 * 60 + 55
    london_afternoon_start = 14 * 60
    london_afternoon_end = 16 * 60 + 55
    
    df['in_trading_window'] = (
        ((df['minute_of_day'] >= london_morning_start) & (df['minute_of_day'] < london_morning_end)) |
        ((df['minute_of_day'] >= london_afternoon_start) & (df['minute_of_day'] < london_afternoon_end))
    )
    
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
    
    high_4h2 = df['high'].shift(2)
    low_4h2 = df['low'].shift(2)
    
    df['is_swing_high'] = (
        (df['high'].shift(1) < high_4h2) &
        (df['high'].shift(3) < high_4h2) &
        (df['high'].shift(4) < high_4h2)
    )
    df['is_swing_low'] = (
        (df['low'].shift(1) > low_4h2) &
        (df['low'].shift(3) > low_4h2) &
        (df['low'].shift(4) > low_4h2)
    )
    
    df['is_swing_high_5m'] = (
        (df['high'].shift(3) < df['high'].shift(2)) &
        (df['high'].shift(1) <= df['high'].shift(2)) &
        (df['high'].shift(2) >= df['high'].shift(4)) &
        (df['high'].shift(2) >= df['high'].shift(5))
    )
    df['is_swing_low_5m'] = (
        (df['low'].shift(3) > df['low'].shift(2)) &
        (df['low'].shift(1) >= df['low'].shift(2)) &
        (df['low'].shift(2) <= df['low'].shift(4)) &
        (df['low'].shift(2) <= df['low'].shift(5))
    )
    
    df['bullish_count'] = 0
    df['bearish_count'] = 0
    df['trend_direction'] = 'Neutral'
    
    for i in range(1, len(df)):
        if df.loc[i, 'is_swing_high_5m']:
            df.loc[i, 'bullish_count'] = df.loc[i-1, 'bullish_count'] + 1
            df.loc[i, 'bearish_count'] = 0
        elif df.loc[i, 'is_swing_low_5m']:
            df.loc[i, 'bearish_count'] = df.loc[i-1, 'bearish_count'] + 1
            df.loc[i, 'bullish_count'] = 0
        else:
            df.loc[i, 'bullish_count'] = df.loc[i-1, 'bullish_count']
            df.loc[i, 'bearish_count'] = df.loc[i-1, 'bearish_count']
        
        if df.loc[i, 'bullish_count'] > 1:
            df.loc[i, 'trend_direction'] = 'Bullish'
        elif df.loc[i, 'bearish_count'] > 1:
            df.loc[i, 'trend_direction'] = 'Bearish'
        else:
            df.loc[i, 'trend_direction'] = df.loc[i-1, 'trend_direction']
    
    df['atr_buffer'] = df['atr'] * 0.5
    
    df['bullish_fvg'] = (
        (df['low'] < low_4h2) &
        (df['close'] > df['low'] - df['atr_buffer'])
    )
    df['bearish_fvg'] = (
        (df['high'] > high_4h2) &
        (df['close'] < df['high'] + df['atr_buffer'])
    )
    
    df['long_condition'] = (
        df['bullish_fvg'] &
        (df['trend_direction'] != 'Bearish') &
        df['in_trading_window']
    )
    df['short_condition'] = (
        df['bearish_fvg'] &
        (df['trend_direction'] != 'Bullish') &
        df['in_trading_window']
    )
    
    entries = []
    trade_num = 1
    
    for i in range(5, len(df)):
        if pd.isna(df['atr'].iloc[i]) or pd.isna(df['trend_direction'].iloc[i]):
            continue
        
        direction = None
        if df['long_condition'].iloc[i]:
            direction = 'long'
        elif df['short_condition'].iloc[i]:
            direction = 'short'
        
        if direction is None:
            continue
        
        ts = int(df['ts'].iloc[i])
        entry_price = float(df['close'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        entries.append({
            'trade_num': trade_num,
            'direction': direction,
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