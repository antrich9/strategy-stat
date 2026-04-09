import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    tp = 0.5
    length = 5
    tradetype = "Long and Short"
    useCloseCandle = False

    h = df['high'].rolling(length * 2 + 1).max()
    l = df['low'].rolling(length * 2 + 1).min()

    entries = []
    trade_num = 1

    lastLow = df['high'].iloc[0] * 100
    lastHigh = 0.0
    dirUp = False

    for i in range(len(df)):
        if i < length * 2 + 1:
            continue
        
        isMin = l.iloc[i] == df['low'].iloc[i]
        isMax = h.iloc[i] == df['high'].iloc[i]
        
        if dirUp:
            if isMin and df['low'].iloc[i] < lastLow:
                lastLow = df['low'].iloc[i]
            if isMax and df['high'].iloc[i] > lastLow:
                lastHigh = df['high'].iloc[i]
                dirUp = False
        else:
            if isMax and df['high'].iloc[i] > lastHigh:
                lastHigh = df['high'].iloc[i]
            if isMin and df['low'].iloc[i] < lastHigh:
                lastLow = df['low'].iloc[i]
                dirUp = True
                if isMax and df['high'].iloc[i] > lastLow:
                    lastHigh = df['high'].iloc[i]
                    dirUp = False

        recentTouch = False
        for j in range(1, 11):
            if i + j < len(df) and i + j + 1 < len(df):
                if (df['low'].iloc[i + j] <= lastLow and df['low'].iloc[i + j + 1] > lastLow) or (df['high'].iloc[i + j] >= lastHigh and df['high'].iloc[i + j + 1] < lastHigh):
                    recentTouch = True
                    break

        if tradetype in ["Long and Short", "Long"]:
            price_check_long = df['close'].iloc[i] if useCloseCandle else df['high'].iloc[i]
            if price_check_long >= lastHigh and df['high'].iloc[i-1] < lastHigh and not recentTouch:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': df['close'].iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': df['close'].iloc[i],
                    'raw_price_b': df['close'].iloc[i]
                })
                trade_num += 1

        if tradetype in ["Long and Short", "Short"]:
            price_check_short = df['close'].iloc[i] if useCloseCandle else df['low'].iloc[i]
            if price_check_short <= lastLow and df['low'].iloc[i-1] > lastLow and not recentTouch:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': df['close'].iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': df['close'].iloc[i],
                    'raw_price_b': df['close'].iloc[i]
                })
                trade_num += 1

    return entries