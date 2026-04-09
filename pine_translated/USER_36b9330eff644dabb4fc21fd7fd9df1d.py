import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Column series
    high = df['high']
    low = df['low']
    close = df['close']

    # Shifted values
    high_1 = high.shift(1)
    high_2 = high.shift(2)
    low_1 = low.shift(1)
    low_2 = low.shift(2)
    close_1 = close.shift(1)

    # Bull/Bear detection
    bullG = low > high_1
    bearG = high < low_1

    # True Range
    tr = np.maximum(high - low, np.maximum(np.abs(high - close_1), np.abs(low - close_1)))

    # Wilder ATR (length 144)
    atr_len = 144
    atr = tr.ewm(alpha=1/atr_len, adjust=False).mean()
    atr = atr.fillna(0.0)

    # Fair Value Gap width filter (default 0.5)
    fvg_th = 0.5
    atr_filter = atr * fvg_th

    # Bull entry condition
    bull = (low - high_2) > atr_filter
    bull = bull & (low > high_2)
    bull = bull & (close_1 > high_2)
    bull = bull & ~(bullG | bullG.shift(1))

    # Bear entry condition
    bear = (low_2 - high) > atr_filter
    bear = bear & (high < low_2)
    bear = bear & (close_1 < low_2)
    bear = bear & ~(bearG | bearG.shift(1))

    # Ensure NaNs become False
    bull = bull.fillna(False)
    bear = bear.fillna(False)

    # Generate entry list
    entries = []
    trade_num = 1

    for i in df.index:
        if bull.iloc[i]:
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
        if bear.iloc[i]:
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