import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Indicator parameters (default values from the Pine Script)
    ma_length = 100
    stiff_length = 60
    stiff_smooth = 3
    threshold = 90
    use_stiffness = False  # default in the original script

    close = df['close'].copy()

    # 1. Compute the stiffness band (SMA - 0.2 * stddev)
    sma = close.rolling(window=ma_length).mean()
    std = close.rolling(window=ma_length).std(ddof=1)   # sample standard deviation
    bound = sma - 0.2 * std

    # 2. Count bars where close > bound over stiff_length bars
    above = (close > bound).astype(int)
    sum_above = above.rolling(window=stiff_length).sum()

    # 3. Normalise to a percentage
    pct_above = sum_above * 100.0 / stiff_length

    # 4. Smooth the percentage with an EMA
    stiffness = pct_above.ewm(span=stiff_smooth, adjust=False).mean()

    # 5. Determine the signal
    if use_stiffness:
        signal = stiffness > threshold
    else:
        signal = pd.Series(True, index=df.index)

    # 6. Entry condition: signal true and stiffness is not NaN (warm‑up period)
    condition = signal & stiffness.notna()

    # 7. Iterate over bars and emit an entry for every qualifying bar
    entries = []
    trade_num = 1
    for i in df.index:
        if condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = close.iloc[i]
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
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

    return entries