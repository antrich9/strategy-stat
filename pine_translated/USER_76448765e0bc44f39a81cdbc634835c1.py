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
    
    # Resample to 240min for EMAs
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df_240 = df.set_index('time_dt').resample('240T').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna(subset=['close'])
    df_240['fastEMA'] = df_240['close'].ewm(span=9, adjust=False).mean()
    df_240['slowEMA'] = df_240['close'].ewm(span=18, adjust=False).mean()
    
    # Merge EMAs back
    ema_df = df_240[['fastEMA', 'slowEMA']].reset_index().rename(columns={'time_dt': 'time_dt_240'})
    df = df.merge(ema_df, left_on='time_dt', right_on='time_dt_240', how='left')
    df['fastEMA'] = df['fastEMA'].ffill()
    df['slowEMA'] = df['slowEMA'].ffill()
    
    # Session filter: London 07:00-10:00 and NY 14:00-17:00 UTC
    hour = df['time_dt'].dt.hour
    isSessionActive = ((hour >= 7) & (hour < 10)) | ((hour >= 14) & (hour < 17))
    
    # Trend conditions
    condition_long = (df['close'] > df['fastEMA']) & (df['close'] > df['slowEMA'])
    condition_short = (df['close'] < df['fastEMA']) & (df['close'] < df['slowEMA'])
    isBullishTrend = (df['close'] > df['slowEMA']) & (df['fastEMA'] > df['slowEMA'])
    isBearishTrend = (df['close'] < df['slowEMA']) & (df['fastEMA'] < df['slowEMA'])
    
    # Filters (using inp defaults as true when not specified)
    volfilt = df['volume'] > df['volume'].shift(1).rolling(9).mean() * 1.5
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    atr_rolling = tr.rolling(20).mean() / 1.5
    atrfilt = (df['low'] - df['high'].shift(2) > atr_rolling) | (df['low'].shift(2) - df['high'] > atr_rolling)
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # FVG conditions
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    # OB conditions
    isDown = df['close'] < df['open']
    isUp = df['close'] > df['open']
    obUp = isDown.shift(1) & isUp & (df['close'] > df['high'].shift(1))
    obDown = isUp.shift(1) & isDown & (df['close'] < df['low'].shift(1))
    
    # Entry conditions
    long_condition = condition_long & isSessionActive & isBullishTrend & (bfvg | obUp)
    short_condition = condition_short & isSessionActive & isBearishTrend & (sfvg | obDown)
    
    # Skip bars where required indicators are NaN
    valid_mask = ~(df['fastEMA'].isna() | df['slowEMA'].isna())
    long_condition = long_condition & valid_mask
    short_condition = short_condition & valid_mask
    
    # Generate entries
    trade_num = 1
    entries = []
    
    for i in range(len(df)):
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