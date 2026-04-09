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
    df['datetime'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['total_minutes'] = df['hour'] * 60 + df['minute']

    is_time_filter_long = (df['total_minutes'] >= 420) & (df['total_minutes'] <= 599)
    is_time_filter_short = (df['total_minutes'] >= 720) & (df['total_minutes'] <= 899)

    df['is_new_day'] = df['datetime'].dt.date.ne(df['datetime'].dt.date.shift(1))

    df['prev_day_high'] = df['high'].shift(1).where(df['is_new_day'].shift(1), np.nan)
    df['prev_day_low'] = df['low'].shift(1).where(df['is_new_day'].shift(1), np.nan)

    for i in range(1, len(df)):
        if df['is_new_day'].iloc[i]:
            df.loc[df.index[i], 'flagpdh'] = False
            df.loc[df.index[i], 'flagpdl'] = False
            df.loc[df.index[i], 'waitingForEntry'] = False
            df.loc[df.index[i], 'waitingForShortEntry'] = False
        else:
            if i > 0:
                df.loc[df.index[i], 'flagpdh'] = df['flagpdh'].iloc[i-1] if 'flagpdh' in df.columns else False
                df.loc[df.index[i], 'flagpdl'] = df['flagpdl'].iloc[i-1] if 'flagpdl' in df.columns else False
                df.loc[df.index[i], 'waitingForEntry'] = df['waitingForEntry'].iloc[i-1] if 'waitingForEntry' in df.columns else False
                df.loc[df.index[i], 'waitingForShortEntry'] = df['waitingForShortEntry'].iloc[i-1] if 'waitingForShortEntry' in df.columns else False

        if df['close'].iloc[i] > df['prev_day_high'].iloc[i]:
            df.loc[df.index[i], 'flagpdh'] = True
        if df['close'].iloc[i] < df['prev_day_low'].iloc[i]:
            df.loc[df.index[i], 'flagpdl'] = True

    if 'flagpdh' not in df.columns:
        df['flagpdh'] = False
    if 'flagpdl' not in df.columns:
        df['flagpdl'] = False
    if 'waitingForEntry' not in df.columns:
        df['waitingForEntry'] = False
    if 'waitingForShortEntry' not in df.columns:
        df['waitingForShortEntry'] = False

    df['is_down_1'] = df['close'].shift(1) < df['open'].shift(1)
    df['is_up_0'] = df['close'] > df['open']
    df['ob_up'] = df['is_down_1'] & df['is_up_0'] & (df['close'] > df['high'].shift(1))

    df['is_up_1'] = df['close'].shift(1) > df['open'].shift(1)
    df['is_down_0'] = df['close'] < df['open']
    df['ob_down'] = df['is_up_1'] & df['is_down_0'] & (df['close'] < df['low'].shift(1))

    df['fvg_up'] = df['low'] > df['high'].shift(2)
    df['fvg_down'] = df['high'] < df['low'].shift(2)

    df['bullish_ob_fvg'] = df['ob_up'] & df['fvg_up']
    df['bearish_ob_fvg'] = df['ob_down'] & df['fvg_down']

    entries = []
    trade_num = 1
    in_position = False
    in_short_position = False

    for i in range(len(df)):
        row = df.iloc[i]

        if row['bullish_ob_fvg'] and is_time_filter_long.iloc[i] and row['flagpdl'] and not in_position:
            in_position = True
            entry_ts = int(row['ts'])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(row['close'])

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

        if row['bearish_ob_fvg'] and is_time_filter_short.iloc[i] and row['flagpdh'] and not in_short_position:
            in_short_position = True
            entry_ts = int(row['ts'])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(row['close'])

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