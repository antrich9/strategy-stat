import pandas as pd
import numpy as np
from datetime import datetime, timezone

def calculate_wilder_atr(high, low, close, length=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def resample_to_4h(df):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['period_4h'] = df['datetime'].dt.floor('4h')
    agg = df.groupby('period_4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'time': 'first'
    }).reset_index(drop=True)
    return agg

def generate_entries(df: pd.DataFrame) -> list:
    if len(df) < 3:
        return []
    df = df.copy()
    agg_4h = resample_to_4h(df)
    high_4h = agg_4h['high']
    low_4h = agg_4h['low']
    close_4h = agg_4h['close']
    volume_4h = agg_4h['volume']
    time_4h = agg_4h['time']
    vol_sma = volume_4h.rolling(9).mean() * 1.5
    volfilt = volume_4h.shift(1) > vol_sma
    atr_4h = calculate_wilder_atr(high_4h, low_4h, close_4h, 20) / 1.5
    atrfilt = ((low_4h - high_4h.shift(2) > atr_4h) | (low_4h.shift(2) - high_4h > atr_4h))
    loc1 = close_4h.rolling(54).mean()
    loc2 = loc1 > loc1.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    bull_fvg = (low_4h > high_4h.shift(2)) & volfilt & atrfilt & locfiltb
    bear_fvg = (high_4h < low_4h.shift(2)) & volfilt & atrfilt & locfilts
    last_fvg = 0
    entries = []
    trade_num = 1
    for i in range(1, len(agg_4h)):
        if bull_fvg.iloc[i] and last_fvg == -1:
            ts = int(time_4h.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close_4h.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_4h.iloc[i]),
                'raw_price_b': float(close_4h.iloc[i])
            })
            trade_num += 1
            last_fvg = 1
        elif bear_fvg.iloc[i] and last_fvg == 1:
            ts = int(time_4h.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close_4h.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_4h.iloc[i]),
                'raw_price_b': float(close_4h.iloc[i])
            })
            trade_num += 1
            last_fvg = -1
        elif bull_fvg.iloc[i]:
            last_fvg = 1
        elif bear_fvg.iloc[i]:
            last_fvg = -1
    return entries