import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    sma20 = close.rolling(20).mean()

    # crossover / crossunder
    long_condition = (close > sma20) & (close.shift(1) <= sma20.shift(1))
    short_condition = (close < sma20) & (close.shift(1) >= sma20.shift(1))

    # extract hour from unix timestamp (assumes UTC)
    hours = pd.to_datetime(df['time'], unit='s', utc=True).dt.hour

    # kill zones: London 3-4 GMT, New York 10-11 GMT
    in_kill_zone = ((hours >= 3) & (hours < 4)) | ((hours >= 10) & (hours < 11))

    long_entry = long_condition & in_kill_zone
    short_entry = short_condition & in_kill_zone

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(sma20.iloc[i]):
            continue

        if long_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1

        elif short_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1

    return entries