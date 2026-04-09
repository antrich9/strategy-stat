import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df_indexed = df.set_index(pd.to_datetime(df['time'], unit='s', utc=True))
    df_4h = df_indexed.resample('4h').last()
    ema9_4h = df_4h['close'].ewm(span=9, adjust=False).mean()
    ema18_4h = df_4h['close'].ewm(span=18, adjust=False).mean()
    ema9 = ema9_4h.reindex(df_indexed.index, method='ffill')
    ema18 = ema18_4h.reindex(df_indexed.index, method='ffill')
    close = df['close']
    condition_long = (close > ema9) & (close > ema18)
    condition_short = (close < ema9) & (close < ema18)
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if condition_long.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif condition_short.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    return entries