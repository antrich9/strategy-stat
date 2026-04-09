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
    # Keep a copy to avoid mutating the input
    df = df.copy()

    # ----- compute previous day's high and low -----
    # Determine the calendar day for each bar
    df['date'] = pd.to_datetime(df['time'], unit='s').dt.date

    # Aggregate daily high and low
    daily_agg = df.groupby('date').agg(
        high=('high', 'max'),
        low=('low', 'min')
    ).reset_index()

    # Shift to obtain the previous day's values
    daily_agg['prev_high'] = daily_agg['high'].shift(1)
    daily_agg['prev_low'] = daily_agg['low'].shift(1)

    # Map previous day's high/low back to each bar
    prev_high_map = daily_agg.set_index('date')['prev_high']
    prev_low_map = daily_agg.set_index('date')['prev_low']
    df['prevDayHigh'] = df['date'].map(prev_high_map)
    df['prevDayLow'] = df['date'].map(prev_low_map)

    # ----- entry condition series -----
    # long: close crosses above previous day's high
    long_condition = (
        (df['close'] > df['prevDayHigh']) &
        (df['close'].shift(1) <= df['prevDayHigh'].shift(1))
    )

    # short: close crosses below previous day's low
    short_condition = (
        (df['close'] < df['prevDayLow']) &
        (df['close'].shift(1) >= df['prevDayLow'].shift(1))
    )

    # ----- generate entry records -----
    entries = []
    trade_num = 1

    for i in range(len(df)):
        # Skip bars where required daily data is missing
        if pd.isna(df['prevDayHigh'].iloc[i]) or pd.isna(df['prevDayLow'].iloc[i]):
            continue

        if long_condition.iloc[i]:
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries