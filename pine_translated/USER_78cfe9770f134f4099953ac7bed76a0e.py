import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    atr_period = 22
    atr_multiplier = 3.0

    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()
    atr_value = atr * atr_multiplier

    upper_band = pd.Series(np.nan, index=df.index, dtype=float)
    lower_band = pd.Series(np.nan, index=df.index, dtype=float)
    direction = pd.Series(np.nan, index=df.index, dtype=float)

    for i in range(len(df)):
        if i == 0:
            direction.iloc[i] = 1
            continue

        if pd.isna(upper_band.iloc[i-1]):
            prev_upper = close.iloc[i-1] - atr_value.iloc[i-1]
        else:
            prev_upper = upper_band.iloc[i-1]

        if pd.isna(lower_band.iloc[i-1]):
            prev_lower = close.iloc[i-1] + atr_value.iloc[i-1]
        else:
            prev_lower = lower_band.iloc[i-1]

        if close.iloc[i] < prev_upper:
            upper_band.iloc[i] = high.iloc[i] - atr_value.iloc[i]
        else:
            upper_band.iloc[i] = np.nan

        if close.iloc[i] > prev_lower:
            lower_band.iloc[i] = low.iloc[i] + atr_value.iloc[i]
        else:
            lower_band.iloc[i] = np.nan

        if pd.isna(direction.iloc[i-1]):
            direction.iloc[i] = 1
        elif direction.iloc[i-1] == -1 and close.iloc[i] > prev_lower:
            direction.iloc[i] = 1
        elif direction.iloc[i-1] == 1 and close.iloc[i] < prev_upper:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]

    entries = []
    trade_num = 1

    for i in range(1, len(df)):
        if pd.isna(direction.iloc[i]) or pd.isna(direction.iloc[i-1]):
            continue

        if direction.iloc[i] == 1 and direction.iloc[i-1] == -1:
            entry_price = close.iloc[i]
            entry_ts = int(df['time'].iloc[i])
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

        if direction.iloc[i] == -1 and direction.iloc[i-1] == 1:
            entry_price = close.iloc[i]
            entry_ts = int(df['time'].iloc[i])
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