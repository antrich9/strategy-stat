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
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    time = df['time']

    # McGinley Dynamic
    length_md = 10
    md = np.zeros(len(df))
    for i in range(len(df)):
        if i == 0:
            md[i] = close.iloc[i]
        else:
            if np.isnan(md[i-1]) or md[i-1] == 0:
                md[i] = close.iloc[i]
            else:
                md[i] = md[i-1] + (close.iloc[i] - md[i-1]) / (length_md * np.power(close.iloc[i] / md[i-1], 4))

    md_series = pd.Series(md, index=df.index)

    # Trendilo Calculation
    smooth = 1
    lengthTrendilo = 50
    offset = 0.85
    sigma = 6
    bmult = 1.0
    cblen = False
    blen = 20

    pch = (close.diff(smooth) / close.shift(smooth)) * 100

    # ALMA approximation
    def alma(arr, length, offset, sigma):
        window = np.arange(length)
        m = offset * (length - 1)
        s = sigma * length / 6.0
        w = np.exp(-np.power(window - m, 2) / (2 * np.power(s, 2)))
        w = w / w.sum()
        return pd.Series(np.convolve(arr, w, mode='valid')).shift(length - 1).reindex(arr.index, fill_value=np.nan)

    avpch = alma(pch.values, lengthTrendilo, offset, sigma)

    blength = blen if cblen else lengthTrendilo

    rms = bmult * np.sqrt(avpch.rolling(blength).apply(lambda x: (x**2).sum() / blength, raw=True))

    cdir = pd.Series(np.where(avpch > rms, 1, np.where(avpch < -rms, -1, 0)), index=df.index)

    # Normalized Volume
    Length_vol = 50
    hv = 150

    nVolume = volume / volume.rolling(Length_vol).mean() * 100

    # Entry Conditions
    longCondition = (cdir == 1) & (nVolume >= hv) & (md_series > md_series.shift(1)) & (close > md_series)
    shortCondition = (cdir == -1) & (nVolume >= hv) & (md_series < md_series.shift(1)) & (close < md_series)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i < blength or np.isnan(md_series.iloc[i]) or np.isnan(avpch.iloc[i]):
            continue

        if longCondition.iloc[i]:
            entry_ts = int(time.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])

            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        elif shortCondition.iloc[i]:
            entry_ts = int(time.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])

            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
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