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
    donchLength = 20
    activateGreenElephantCandles = True
    activateRedElephantCandles = True
    minBodyPercentage = 70
    previousBarsCount = 100
    searchFactor = 1.3
    smooth = 1
    length = 50
    offset = 0.85
    sigma = 6
    bmult = 1.0
    cblen = False
    blen = 20
    atrLen = 100

    open_col = df['open']
    high_col = df['high']
    low_col = df['low']
    close_col = df['close']

    highestHigh = high_col.rolling(window=donchLength).max()
    lowestLow = low_col.rolling(window=donchLength).min()

    atr_val = np.zeros(len(df))
    prev_atr = 0.0
    tr = np.maximum(high_col.diff().abs().fillna(0), np.maximum(high_col - low_col, (high_col - close_col.shift(1)).abs().fillna(0)))
    for i in range(len(df)):
        if i < atrLen - 1:
            atr_val[i] = np.nan
        elif i == atrLen - 1:
            atr_val[i] = tr.iloc[:atrLen].mean()
            prev_atr = atr_val[i]
        else:
            prev_atr = prev_atr * (atrLen - 1) / atrLen + tr.iloc[i]
            atr_val[i] = prev_atr
    atr_series = pd.Series(atr_val, index=df.index)

    body = (close_col - open_col).abs()
    range_abc = high_col - low_col
    isGreenElephantCandle = close_col > open_col
    isRedElephantCandle = close_col < open_col
    body_pct = body * 100 / range_abc.replace(0, np.nan)
    isGreenElephantCandleValid = isGreenElephantCandle & (body_pct >= minBodyPercentage)
    isRedElephantCandleValid = isRedElephantCandle & (body_pct >= minBodyPercentage)
    isGreenElephantCandleStrong = isGreenElephantCandleValid & (body >= atr_series.shift(1) * searchFactor)
    isRedElephantCandleStrong = isRedElephantCandleValid & (body >= atr_series.shift(1) * searchFactor)

    src = close_col
    pch = src.pct_change(smooth) / src * 100
    m = offset * (length - 1)
    s = sigma * length / 6
    alma_weights = np.exp(-(np.arange(length) - m) ** 2 / (2 * s * s))
    alma_weights = alma_weights / alma_weights.sum()
    avpch = pch.rolling(window=length).apply(lambda x: (x * alma_weights).sum() if len(x) == length else np.nan, raw=True)
    blength = blen if cblen else length
    rms = bmult * np.sqrt(avpch.rolling(window=blength).apply(lambda x: (x * x).sum() / blength, raw=True))

    cdir_vals = np.where(avpch > rms, 1, np.where(avpch < -rms, -1, 0))
    cdir_series = pd.Series(cdir_vals, index=df.index)

    has_position = False
    trade_num = 1
    entries = []

    for i in range(len(df)):
        if has_position:
            continue
        if i < max(donchLength, atrLen, length, blength) - 1:
            continue
        if pd.isna(atr_series.iloc[i]) or pd.isna(avpch.iloc[i]) or pd.isna(rms.iloc[i]):
            continue
        if pd.isna(highestHigh.iloc[i]) or pd.isna(lowestLow.iloc[i]):
            continue
        if cdir_series.iloc[i] > 0 and activateGreenElephantCandles and isGreenElephantCandleStrong.iloc[i]:
            entry_price = close_col.iloc[i]
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            raw_price_a = entry_price
            raw_price_b = entry_price
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': raw_price_a,
                'raw_price_b': raw_price_b
            })
            trade_num += 1
            has_position = True
        elif cdir_series.iloc[i] < 0 and activateRedElephantCandles and isRedElephantCandleStrong.iloc[i]:
            entry_price = close_col.iloc[i]
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            raw_price_a = entry_price
            raw_price_b = entry_price
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': raw_price_a,
                'raw_price_b': raw_price_b
            })
            trade_num += 1
            has_position = True

    return entries