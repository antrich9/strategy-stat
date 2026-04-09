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
    df = df.copy()
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['dt'].dt.hour
    df['dayofweek'] = df['dt'].dt.dayofweek

    n1 = 10
    n2 = 21
    obLevel2 = 53
    osLevel2 = -53

    hlc3 = (df['high'] + df['low'] + df['close']) / 3

    esa = hlc3.ewm(span=n1, adjust=False).mean()
    d = (hlc3 - esa).abs().ewm(span=n1, adjust=False).mean()
    ci = (hlc3 - esa) / (0.015 * d)
    tci = ci.ewm(span=n2, adjust=False).mean()
    wt1 = tci
    wt2 = wt1.rolling(4).mean()

    wt1_H = wt1.copy()
    wt2_H = wt2.copy()
    wt1_D = wt1.copy()
    wt2_D = wt2.copy()

    crossUp_H = (wt1_H > wt2_H) & (wt1_H.shift(1) <= wt2_H.shift(1))
    crossDown_H = (wt1_H < wt2_H) & (wt1_H.shift(1) >= wt2_H.shift(1))
    crossUp_D = (wt1_D > wt2_D) & (wt1_D.shift(1) <= wt2_D.shift(1))
    crossDown_D = (wt1_D < wt2_D) & (wt1_D.shift(1) >= wt2_D.shift(1))

    isOversold_H = wt2_H < osLevel2
    isOverbought_H = wt2_H > obLevel2
    isOversold_D = wt2_D < osLevel2
    isOverbought_D = wt2_D > obLevel2

    longCondition = crossUp_H & isOversold_H & isOversold_D
    shortCondition = crossDown_H & isOverbought_H & isOverbought_D

    morning_window = (df['hour'] >= 8) & (df['hour'] < 10)
    afternoon_window = (df['hour'] >= 14) & (df['hour'] < 17)
    isFridayMorningWindow = (df['dayofweek'] == 4) & morning_window
    in_trading_window = (morning_window | afternoon_window) & (~isFridayMorningWindow)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(wt1_H.iloc[i]) or pd.isna(wt1_D.iloc[i]):
            continue

        if in_trading_window.iloc[i]:
            if longCondition.iloc[i]:
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
            elif shortCondition.iloc[i]:
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