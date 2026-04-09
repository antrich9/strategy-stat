import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Compute EMAs
    close = df['close']
    ema_short = close.ewm(span=10, adjust=False).mean()
    ema_long = close.ewm(span=20, adjust=False).mean()

    # Crossover / crossunder signals
    crossover = (ema_short > ema_long) & (ema_short.shift(1) <= ema_long.shift(1))
    crossunder = (ema_short < ema_long) & (ema_short.shift(1) >= ema_long.shift(1))

    # Price relative to EMAs
    cond_above = (close > ema_short) & (close > ema_long)
    cond_below = (close < ema_short) & (close < ema_long)

    # Base entry conditions
    bull = crossover & cond_above
    bear = crossunder & cond_below

    # Wilder RSI (14‑period)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)

    # Apply RSI filter (use_rsi is hard‑coded true)
    bull = bull & (rsi < 30)
    bear = bear & (rsi > 70)

    # Mask out bars with NaN in any required indicator
    mask = (bull | bear) & ema_short.notna() & ema_long.notna() & rsi.notna()

    entries = []
    trade_num = 1
    for idx in df[mask].index:
        ts = int(df.loc[idx, 'time'])
        price = float(df.loc[idx, 'close'])
        direction = 'long' if bull.loc[idx] else 'short'
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entries.append({
            'trade_num': trade_num,
            'direction': direction,
            'entry_ts': ts,
            'entry_time': entry_time,
            'entry_price_guess': price,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': price,
            'raw_price_b': price
        })
        trade_num += 1

    return entries