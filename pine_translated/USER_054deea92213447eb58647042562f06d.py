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
    # work on a copy to avoid mutating input
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Time & timezone handling
    # ------------------------------------------------------------------
    # ensure time is integer seconds
    df['time'] = df['time'].astype(int)

    # UTC datetime
    df['datetime_utc'] = pd.to_datetime(df['time'], unit='s', utc=True)

    # convert to London time (Europe/London)
    df['datetime'] = df['datetime_utc'].dt.tz_convert('Europe/London')

    # extract hour/minute for window checks
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute

    # ------------------------------------------------------------------
    # 2. Define trading windows (London time)
    #    Morning  08:00 – 09:45
    #    Afternoon 16:00 – 16:45
    # ------------------------------------------------------------------
    morning_window = (df['hour'] == 8) | ((df['hour'] == 9) & (df['minute'] <= 45))
    afternoon_window = (df['hour'] == 16) & (df['minute'] <= 45)
    df['isWithinTimeWindow'] = morning_window | afternoon_window

    # ------------------------------------------------------------------
    # 3. Previous day high / low (daily OHLC)
    # ------------------------------------------------------------------
    df['date'] = df['datetime'].dt.date
    daily_agg = df.groupby('date').agg(daily_high=('high', 'max'), daily_low=('low', 'min')).reset_index()
    # shift daily data to obtain previous day's high/low
    daily_agg['prevDayHigh'] = daily_agg['daily_high'].shift(1)
    daily_agg['prevDayLow'] = daily_agg['daily_low'].shift(1)

    # merge previous day values back onto the minute bars
    df = df.merge(daily_agg[['date', 'prevDayHigh', 'prevDayLow']], on='date', how='left')

    # ------------------------------------------------------------------
    # 4. Approximate current 4‑hour high / low (rolling window)
    #    Using a fixed 240‑period window – assumes minute‑level data.
    # ------------------------------------------------------------------
    df['currentDayLow'] = df['low'].rolling(window=240, min_periods=1).min()
    df['currentDayHigh'] = df['high'].rolling(window=240, min_periods=1).max()

    # ------------------------------------------------------------------
    # 5. Sweep conditions
    # ------------------------------------------------------------------
    df['prevDayHighTaken'] = df['high'] > df['prevDayHigh']
    df['prevDayLowTaken'] = df['low'] < df['prevDayLow']

    # flagpdh: previous day high swept AND current 4‑h low above prev day low
    df['flagpdh'] = df['prevDayHighTaken'] & (df['currentDayLow'] > df['prevDayLow'])
    # flagpdl: previous day low swept AND current 4‑h high below prev day high
    df['flagpdl'] = df['prevDayLowTaken'] & (df['currentDayHigh'] < df['prevDayHigh'])

    # ------------------------------------------------------------------
    # 6. Valid rows (no NaN in required indicators)
    # ------------------------------------------------------------------
    valid = (df['prevDayHigh'].notna() &
             df['prevDayLow'].notna() &
             df['currentDayLow'].notna() &
             df['currentDayHigh'].notna())

    # ------------------------------------------------------------------
    # 7. Detect first occurrence of each flag (crossover)
    # ------------------------------------------------------------------
    prev_flagpdh = df['flagpdh'].shift(1).fillna(False)
    prev_flagpdl = df['flagpdl'].shift(1).fillna(False)

    entry_long = df['flagpdh'] & ~prev_flagpdh & df['isWithinTimeWindow'] & valid
    entry_short = df['flagpdl'] & ~prev_flagpdl & df['isWithinTimeWindow'] & valid

    # ------------------------------------------------------------------
    # 8. Build entry list
    # ------------------------------------------------------------------
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if entry_long.iloc[i] or entry_short.iloc[i]:
            direction = 'long' if entry_long.iloc[i] else 'short'
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])

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