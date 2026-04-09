import pandas as pd
import numpy as np
from datetime import datetime, timezone

def calculate_adx(df, dilen=14, adxlen=14):
    up = df['high'].diff()
    down = -df['low'].diff()

    plusDM = np.where((up > down) & (up > 0), up, 0)
    minusDM = np.where((down > up) & (down > 0), down, 0)

    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift(1))
    tr3 = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.Series(np.maximum(np.maximum(tr1, tr2), tr3), index=df.index)

    truerange = tr.ewm(alpha=1/dilen, adjust=False).mean()
    plus = (pd.Series(plusDM, index=df.index).ewm(alpha=1/dilen, adjust=False).mean() / truerange).fillna(0) * 100
    minus = (pd.Series(minusDM, index=df.index).ewm(alpha=1/dilen, adjust=False).mean() / truerange).fillna(0) * 100

    diff = np.abs(plus - minus)
    sum_dm = plus + minus
    adx_val = (diff / sum_dm.replace(0, 1)).ewm(alpha=1/adxlen, adjust=False).mean() * 100

    return adx_val

def generate_entries(df: pd.DataFrame) -> list:
    jaw = df['close'].ewm(span=13, adjust=False).mean()
    teeth = df['close'].ewm(span=8, adjust=False).mean()
    lips = df['close'].ewm(span=5, adjust=False).mean()
    adx_val = calculate_adx(df)

    entries = []
    trade_num = 1

    for i in range(1, len(df)):
        if pd.isna(jaw.iloc[i]) or pd.isna(teeth.iloc[i]) or pd.isna(lips.iloc[i]) or pd.isna(adx_val.iloc[i]):
            continue

        longCondition = lips.iloc[i] > teeth.iloc[i] and teeth.iloc[i] > jaw.iloc[i] and df['close'].iloc[i] > lips.iloc[i] and adx_val.iloc[i] > 30
        shortCondition = lips.iloc[i] < teeth.iloc[i] and teeth.iloc[i] < jaw.iloc[i] and df['close'].iloc[i] < lips.iloc[i] and adx_val.iloc[i] > 30

        if longCondition:
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

        if shortCondition:
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