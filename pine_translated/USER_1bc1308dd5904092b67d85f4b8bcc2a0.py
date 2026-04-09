import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Compute HTF close (placeholder: using shift(1) as proxy)
    htf_close = df['close'].shift(1)

    # Compute entry conditions
    long_condition = (df['close'] > htf_close) & (df['close'].shift(1) <= htf_close.shift(1))
    short_condition = (df['close'] < htf_close) & (df['close'].shift(1) >= htf_close.shift(1))

    # Initialize list
    entries = []
    trade_num = 1

    # Iterate over bars
    for i in range(1, len(df)):
        # Skip if condition is false or NaN
        if pd.isna(long_condition.iloc[i]) or pd.isna(short_condition.iloc[i]):
            continue

        if long_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])

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

        if short_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])

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