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
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)

    # Daily bars
    daily = df.set_index('dt').resample('1D').agg({'close': 'last'}).dropna()
    ema9 = daily['close'].ewm(span=9, adjust=False).mean()
    ema18 = daily['close'].ewm(span=18, adjust=False).mean()

    df['date'] = df['dt'].dt.date
    ema9_map = ema9.rename(lambda x: x.date()).to_dict()
    ema18_map = ema18.rename(lambda x: x.date()).to_dict()
    df['ema9'] = df['date'].map(ema9_map)
    df['ema18'] = df['date'].map(ema18_map)

    # Europe/London time window
    df['lt'] = df['dt'].dt.tz_convert('Europe/London')
    hour = df['lt'].dt.hour
    minute = df['lt'].dt.minute

    morning = ((hour > 7) | ((hour == 7) & (minute >= 45))) & ((hour < 9) | ((hour == 9) & (minute < 45)))
    afternoon = ((hour > 14) | ((hour == 14) & (minute >= 45))) & ((hour < 16) | ((hour == 16) & (minute < 45)))
    in_window = morning | afternoon

    cond_long = in_window & (df['close'] > df['ema9']) & (df['close'] > df['ema18'])
    cond_short = in_window & (df['close'] < df['ema9']) & (df['close'] < df['ema18'])

    entries = []
    trade_num = 1
    for i in range(len(df)):
        if pd.isna(df['ema9'].iloc[i]) or pd.isna(df['ema18'].iloc[i]):
            continue
        if cond_long.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif cond_short.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    return entries