import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.sort_values('datetime').reset_index(drop=True)
    hour = df['datetime'].dt.hour
    minute = df['datetime'].dt.minute
    morning_start = (hour == 7) & (minute >= 45)
    morning_window = morning_start | (hour == 8) | ((hour == 9) & (minute <= 45))
    afternoon_start = (hour == 15) & (minute >= 45)
    afternoon_window = afternoon_start | (hour == 16)
    in_trading_window = morning_window | afternoon_window
    df['rolling_240_high'] = df['high'].rolling(240).max()
    df['rolling_240_low'] = df['low'].rolling(240).min()
    daily_df = df.set_index('datetime').resample('D').agg({'high': 'max', 'low': 'min'}).reset_index()
    daily_df['prev_day_high'] = daily_df['high'].shift(1)
    daily_df['prev_day_low'] = daily_df['low'].shift(1)
    daily_df = daily_df.dropna(subset=['prev_day_high', 'prev_day_low'])
    daily_df = daily_df[['datetime', 'prev_day_high', 'prev_day_low']]
    daily_df = daily_df.rename(columns={'datetime': 'date'})
    df['date'] = df['datetime'].dt.date
    daily_df['date'] = daily_df['datetime'].dt.date
    df = df.merge(daily_df[['date', 'prev_day_high', 'prev_day_low']], on='date', how='left')
    df['prev_day_high_taken'] = df['high'] > df['prev_day_high']
    df['prev_day_low_taken'] = df['low'] < df['prev_day_low']
    flagpdh = pd.Series(False, index=df.index)
    flagpdl = pd.Series(False, index=df.index)
    for i in range(1, len(df)):
        if pd.isna(df['prev_day_high'].iloc[i]):
            continue
        if df['prev_day_high_taken'].iloc[i] and df['rolling_240_low'].iloc[i] > df['prev_day_low'].iloc[i]:
            flagpdh.iloc[i] = True
        elif df['prev_day_low_taken'].iloc[i] and df['rolling_240_high'].iloc[i] < df['prev_day_high'].iloc[i]:
            flagpdl.iloc[i] = True
    bull_fvg_top = np.minimum(df['close'], df['open'])
    bull_fvg_btm = np.maximum(df['close'].shift(1), df['open'].shift(1))
    bull_fvg = (df['open'] > df['close'].shift(1)) & \
               (df['high'].shift(1) > df['low']) & \
               (df['close'] > df['close'].shift(1)) & \
               (df['open'] > df['open'].shift(1)) & \
               (df['high'].shift(1) < bull_fvg_top)
    bear_fvg_top = np.minimum(df['close'].shift(1), df['open'].shift(1))
    bear_fvg_btm = np.maximum(df['close'], df['open'])
    bear_fvg = (df['close'] < df['open'].shift(1)) & \
               (df['low'].shift(1) < df['high']) & \
               (df['close'] < df['close'].shift(1)) & \
               (df['open'] < df['open'].shift(1)) & \
               (df['low'].shift(1) > bear_fvg_top)
    long_entry = flagpdh & in_trading_window & bull_fvg
    short_entry = flagpdl & in_trading_window & bear_fvg
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if long_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    return entries