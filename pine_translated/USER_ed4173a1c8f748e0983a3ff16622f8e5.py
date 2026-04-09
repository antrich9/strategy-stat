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
    rangeBars = 200
    rows = 50
    useClose = True

    price = df['close'] if useClose else (df['high'] + df['low']) / 2

    entries = []
    trade_num = 1

    n = len(df)

    for i in range(n):
        if i < rangeBars:
            continue

        startIdx = i - rangeBars + 1
        endIdx = i

        range_high = df['high'].iloc[startIdx:endIdx + 1].max()
        range_low = df['low'].iloc[startIdx:endIdx + 1].min()

        hasRange = range_high != range_low

        if not hasRange:
            continue

        prRange = range_high - range_low
        step = prRange / rows
        if step == 0.0:
            step = 1e-7

        volBins = np.zeros(rows)
        priceBins = np.zeros(rows)

        for j in range(rows):
            priceBins[j] = range_low + (j + 0.5) * step

        for j in range(startIdx, endIdx + 1):
            idx = int(np.floor((price.iloc[j] - range_low) / step))
            idx = max(0, min(idx, rows - 1))
            volBins[idx] += df['volume'].iloc[j]

        pocIdx = np.argmax(volBins)
        pocPrice = priceBins[pocIdx]

        if i > 0:
            crossUp = df['close'].iloc[i] > pocPrice and df['close'].iloc[i - 1] <= pocPrice
            crossDown = df['close'].iloc[i] < pocPrice and df['close'].iloc[i - 1] >= pocPrice

            if crossUp:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                })
                trade_num += 1

            if crossDown:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                })
                trade_num += 1

    return entries