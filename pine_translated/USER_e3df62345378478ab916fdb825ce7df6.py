import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Compute EMAs
    slowEma = df['close'].ewm(span=200, adjust=False).mean()
    fastEma = df['close'].ewm(span=50, adjust=False).mean()

    # Trend flags
    upTrend = (df['close'] > slowEma) & (df['close'] > fastEma)
    downTrend = (df['close'] < slowEma) & (df['close'] < fastEma)

    # Candlestick components
    open_s = df['open']
    close_s = df['close']
    high_s = df['high']
    low_s = df['low']

    prev_open = open_s.shift(1)
    prev_close = close_s.shift(1)

    # Engulfing patterns
    bullishEngulfing = (close_s > open_s) & (prev_close < prev_open) & (open_s < prev_close)
    bearishEngulfing = (close_s < open_s) & (prev_close > prev_open) & (open_s > prev_close)

    # Pin bar components
    max_oc = pd.concat([open_s, close_s], axis=1).max(axis=1)
    min_oc = pd.concat([open_s, close_s], axis=1).min(axis=1)
    upperWick = high_s - max_oc
    lowerWick = min_oc - low_s
    bodyLength = (open_s - close_s).abs()

    bullishPinBar = (lowerWick > bodyLength * 2) & (upperWick < bodyLength)
    bearishPinBar = (upperWick > bodyLength * 2) & (lowerWick < bodyLength)

    # Entry conditions
    longCondition = upTrend & (bullishEngulfing | bullishPinBar)
    shortCondition = downTrend & (bearishEngulfing | bearishPinBar)

    entries = []
    trade_num = 1

    n = len(df)
    for i in range(n):
        # Skip bars where required indicators are NaN
        if pd.isna(slowEma.iloc[i]) or pd.isna(fastEma.iloc[i]) or pd.isna(upTrend.iloc[i]) or pd.isna(downTrend.iloc[i]):
            continue

        if longCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        if shortCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries