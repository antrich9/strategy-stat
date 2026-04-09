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
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    df['time_val'] = df['hour'] * 60 + df['minute']

    time_filter = (
        ((df['time_val'] >= 7 * 60) & (df['time_val'] <= 9 * 60 + 59)) |
        ((df['time_val'] >= 12 * 60) & (df['time_val'] <= 14 * 60 + 59))
    )

    daily_hl = df.set_index('dt').resample('D')[['high', 'low']].last()
    daily_hl['prev_day_high'] = daily_hl['high'].shift(1)
    daily_hl['prev_day_low'] = daily_hl['low'].shift(1)
    df = df.join(daily_hl[['prev_day_high', 'prev_day_low']], on='dt')

    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']

    ob_up = (close.shift(2) < open_.shift(2)) & (close.shift(1) > open_.shift(1)) & (close > high.shift(2))
    ob_down = (close.shift(2) > open_.shift(2)) & (close.shift(1) < open_.shift(1)) & (close < low.shift(2))

    fvg_up = low > high.shift(2)
    fvg_down = high < low.shift(2)

    sweep_lo = (close > df['prev_day_low']) & (close.shift(1) <= df['prev_day_low'].shift(1))
    sweep_hi = (close > df['prev_day_high']) & (close.shift(1) <= df['prev_day_high'].shift(1))

    long_cond = ob_up & fvg_up & sweep_lo & time_filter
    short_cond = ob_down & fvg_down & sweep_hi & time_filter

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if long_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': df['dt'].iloc[i].isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': df['dt'].iloc[i].isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1

    return entries