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
    n = len(df)
    entries = []
    trade_num = 1

    open_prices = df['open'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values

    bullish_fvg = pd.Series(False, index=df.index)
    bearish_fvg = pd.Series(False, index=df.index)

    for i in range(2, n):
        if high_prices[i-2] < low_prices[i]:
            bullish_fvg.iloc[i] = True
        if low_prices[i-2] > high_prices[i]:
            bearish_fvg.iloc[i] = True

    ig_active = False
    ig_direction = 0
    ig_c1_high = 0.0
    ig_c1_low = 0.0
    ig_c3_high = 0.0
    ig_c3_low = 0.0
    ig_validation_end = 0

    for i in range(2, n):
        if bullish_fvg.iloc[i]:
            ig_active = True
            ig_direction = 1
            ig_c1_high = high_prices[i-2]
            ig_c1_low = low_prices[i-2]
            ig_c3_high = high_prices[i]
            ig_c3_low = low_prices[i]
            ig_validation_end = i + 4

        if bearish_fvg.iloc[i]:
            ig_active = True
            ig_direction = -1
            ig_c1_high = high_prices[i-2]
            ig_c1_low = low_prices[i-2]
            ig_c3_high = high_prices[i]
            ig_c3_low = low_prices[i]
            ig_validation_end = i + 4

        validated = False
        if ig_active and i <= ig_validation_end:
            if ig_direction == 1 and close_prices[i] < ig_c1_high:
                validated = True
            if ig_direction == -1 and close_prices[i] > ig_c1_low:
                validated = True

        system_entry_long = validated and ig_direction == -1
        system_entry_short = validated and ig_direction == 1

        if system_entry_long:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = close_prices[i]

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
            ig_active = False

        if system_entry_short:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = close_prices[i]

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
            ig_active = False

    return entries