import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # compute EMAs
    ema9 = df['close'].ewm(span=9, adjust=False).mean()
    ema18 = df['close'].ewm(span=18, adjust=False).mean()
    # condition series
    cond_long = (df['close'] > ema9) & (df['close'] > ema18)
    cond_short = (df['close'] < ema9) & (df['close'] < ema18)
    # shift for previous bar
    cond_long_prev = cond_long.shift(1)
    cond_short_prev = cond_short.shift(1)
    # crossover detection (true when current true and previous false or NaN)
    cross_long = cond_long & ((~cond_long_prev) | cond_long_prev.isna())
    cross_short = cond_short & ((~cond_short_prev) | cond_short_prev.isna())
    # combine
    cross_long = cross_long.fillna(False)
    cross_short = cross_short.fillna(False)
    # iterate over bars
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if cross_long.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
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
        elif cross_short.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
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