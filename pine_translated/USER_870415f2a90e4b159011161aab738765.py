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

    # ── timestamps → datetime ──────────────────────────────────────────────────
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['day'] = df['dt'].dt.date

    # ── previous day high / low ────────────────────────────────────────────────
    daily = df.groupby('day').agg({'high': 'max', 'low': 'min'})
    daily = daily.shift(1).rename(columns={'high': 'prevDayHigh', 'low': 'prevDayLow'})
    df = df.join(daily, on='day', how='left')

    # ── current day running high / low ────────────────────────────────────────
    df['currDayHigh'] = df.groupby('day')['high'].cummax()
    df['currDayLow'] = df.groupby('day')['low'].cummin()

    # ── sweep conditions ────────────────────────────────────────────────────────
    df['prevDayHighTaken'] = df['high'] > df['prevDayHigh']
    df['prevDayLowTaken'] = df['low'] < df['prevDayLow']

    df['flagpdh'] = df['prevDayHighTaken'] & (df['currDayLow'] > df['prevDayLow'])
    df['flagpdl'] = df['prevDayLowTaken'] & (df['currDayHigh'] < df['prevDayHigh'])

    # ── trading windows (UTC with UK DST) ──────────────────────────────────────
    def dst_adjusted_hour(ts: pd.Timestamp) -> int:
        month, day, weekday = ts.month, ts.day, ts.weekday()
        if (month > 3 and month < 10) or \
           (month == 3 and weekday == 6 and day >= 25) or \
           (month == 10 and weekday == 6 and day < 25):
            return (ts.hour + 1) % 24
        return ts.hour

    df['hour_adj'] = df['dt'].apply(dst_adjusted_hour)
    df['minute'] = df['dt'].dt.minute

    df['in_trading_window'] = (
        ((df['hour_adj'] >= 7) & (df['hour_adj'] <= 10)) |
        ((df['hour_adj'] >= 15) & (df['hour_adj'] <= 16))
    )

    # ── generate entries ───────────────────────────────────────────────────────
    entries = []
    trade_num = 1

    for i, row in df.iterrows():
        if pd.isna(row['prevDayHigh']) or pd.isna(row['prevDayLow']):
            continue
        if not row['in_trading_window']:
            continue

        direction = None
        if row['flagpdh']:
            direction = 'short'
        elif row['flagpdl']:
            direction = 'long'

        if direction is None:
            continue

        entry_ts = int(row['time'])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        entry_price = float(row['close'])

        entries.append({
            'trade_num': trade_num,
            'direction': direction,
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