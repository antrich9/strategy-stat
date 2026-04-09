import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Ensure required columns
    df = df[['time','open','high','low','close','volume']].copy()
    # Convert time to datetime
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    # Day for grouping
    df['day'] = df['datetime'].dt.date
    # Daily high/low
    daily = df.groupby('day').agg(daily_high=('high','max'), daily_low=('low','min')).reset_index()
    df = df.merge(daily, on='day', how='left')
    # Previous day high/low
    df['prevDayHigh'] = df['daily_high'].shift(1)
    df['prevDayLow'] = df['daily_low'].shift(1)
    # Flags for sweep
    df['flagpdh'] = df['close'] > df['prevDayHigh']
    df['flagpdl'] = df['close'] < df['prevDayLow']
    # Up/Down candles
    df['up'] = df['close'] > df['open']
    df['down'] = df['close'] < df['open']
    # Order Block detection
    df['isObUp'] = df['down'].shift(2) & df['up'].shift(1) & (df['close'].shift(1) > df['high'].shift(2))
    df['isObDown'] = df['up'].shift(2) & df['down'].shift(1) & (df['close'].shift(1) < df['low'].shift(2))
    # Fair Value Gap detection
    df['isFvgUp'] = df['low'] > df['high'].shift(2)
    df['isFvgDown'] = df['high'] < df['low'].shift(2)
    # Stacked OB+FVG
    df['bullStack'] = df['isObUp'] & df['isFvgUp']
    df['bearStack'] = df['isObDown'] & df['isFvgDown']
    # Session time (HHMM)
    df['session'] = df['datetime'].dt.hour * 100 + df['datetime'].dt.minute
    # Morning and afternoon sessions (assuming local time; ignoring timezone)
    df['inSession'] = ((df['session'] >= 700) & (df['session'] <= 959)) | ((df['session'] >= 1200) & (df['session'] <= 1459))
    # Drop rows where needed values are NaN
    df = df.dropna(subset=['prevDayHigh','prevDayLow','bullStack','bearStack'])
    # Reset index for iteration
    df = df.reset_index(drop=True)
    # Iterate to generate entries
    entries = []
    trade_num = 1
    waiting_long = False
    waiting_short = False
    for i in range(len(df)):
        row = df.iloc[i]
        # If a sweep occurs, set waiting flags (if not already waiting)
        if row['flagpdh'] and not waiting_long:
            waiting_long = True
        if row['flagpdl'] and not waiting_short:
            waiting_short = True
        # Long entry condition
        if waiting_long and row['bullStack'] and row['inSession']:
            entry_ts = int(row['time'])
            entry_price = float(row['close'])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
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
            waiting_long = False
        # Short entry condition
        if waiting_short and row['bearStack'] and row['inSession']:
            entry_ts = int(row['time'])
            entry_price = float(row['close'])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
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
            waiting_short = False
    return entries