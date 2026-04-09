import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']

    # shifted series for double‑top/bottom detection
    high_s1 = high.shift(1)
    high_s2 = high.shift(2)
    high_s3 = high.shift(3)
    high_s4 = high.shift(4)
    low_s1 = low.shift(1)
    low_s2 = low.shift(2)
    low_s3 = low.shift(3)
    low_s4 = low.shift(4)

    # double top
    p1_top = (high_s4 < high_s3) & (high_s3 < high_s2) & (high_s2 > high_s1) & (high_s1 < high)
    p2_top = (high_s2 > high) & (np.abs(high_s2 - high) <= high * 0.01)
    doubleTop = (p1_top & p2_top).fillna(False)

    # double bottom
    p1_bot = (low_s4 > low_s3) & (low_s3 > low_s2) & (low_s2 < low_s1) & (low_s1 > low)
    p2_bot = (low_s2 < low) & (np.abs(low_s2 - low) <= low * 0.01)
    doubleBottom = (p1_bot & p2_bot).fillna(False)

    # retracement level
    retracementLevel = 0.618

    # retracement top (only where doubleTop is true)
    retracementTop = pd.Series(
        np.where(doubleTop, high - (high - low_s2) * retracementLevel, np.nan),
        index=df.index
    )

    # retracement bottom (only where doubleBottom is true)
    retracementBottom = pd.Series(
        np.where(doubleBottom, low + (high_s2 - low) * retracementLevel, np.nan),
        index=df.index
    )

    # crossover / crossunder
    crossover = (close > retracementBottom) & (close.shift(1) <= retracementBottom.shift(1))
    crossunder = (close < retracementTop) & (close.shift(1) >= retracementTop.shift(1))

    # signals
    longSignal = (crossover & doubleBottom).fillna(False)
    shortSignal = (crossunder & doubleTop).fillna(False)

    # generate entry list
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if longSignal.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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
        elif shortSignal.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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