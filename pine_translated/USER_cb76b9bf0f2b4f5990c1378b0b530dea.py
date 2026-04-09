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
    # ---------- indicator parameters ----------
    mc_len = 10                     # McGinley length
    trendilo_len = 50               # Trendilo lookback length
    trendilo_offset = 0.85          # ALMA offset
    trendilo_sigma = 6.0            # ALMA sigma
    bmult = 1.0                     # band multiplier
    vol_len = 50                    # volume SMA length
    hv_threshold = 150.0            # high volume threshold

    # ---------- helpers ----------
    def alma(src: pd.Series, length: int, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
        m = offset * (length - 1)
        sigma_factor = sigma * length / 100.0
        k = np.arange(length)
        w = np.exp(-((k - m) ** 2) / (2 * (sigma_factor ** 2)))
        w /= w.sum()
        def weighted_sum(x):
            return (x * w).sum()
        return src.rolling(window=length).apply(weighted_sum, raw=True)

    # ---------- compute indicators ----------
    price = df['close']
    volume = df['volume']

    # McGinley Dynamic
    mc_sma = price.rolling(window=mc_len, min_periods=mc_len).mean()
    md = pd.Series(np.nan, index=price.index, dtype=float)
    # initialise first bar
    if not mc_sma.iloc[0] is np.nan:
        md.iloc[0] = mc_sma.iloc[0]
    else:
        md.iloc[0] = price.iloc[0]
    for i in range(1, len(price)):
        prev_md = md.iloc[i - 1]
        cur_price = price.iloc[i]
        if pd.isna(prev_md):
            cur_sma = mc_sma.iloc[i]
            if pd.isna(cur_sma):
                md.iloc[i] = cur_price
            else:
                md.iloc[i] = cur_sma
        else:
            ratio = cur_price / prev_md
            md.iloc[i] = prev_md + (cur_price - prev_md) / (mc_len * (ratio ** 4))

    # Trendilo
    smooth = 1
    pch = (price - price.shift(smooth)) / price * 100.0
    avpch = alma(pch, trendilo_len, offset=trendilo_offset, sigma=trendilo_sigma)
    # RMS band
    blen = trendilo_len
    sum_sq = (avpch ** 2).rolling(window=blen).sum()
    rms = bmult * np.sqrt(sum_sq / blen)
    cdir = pd.Series(np.where(avpch > rms, 1, np.where(avpch < -rms, -1, 0)), index=avpch.index)

    # Normalised volume
    vol_sma = volume.rolling(window=vol_len).mean()
    nVolume = volume / vol_sma * 100.0

    # ---------- generate entries ----------
    entries = []
    trade_num = 1

    for i in range(len(df)):
        # skip bars with insufficient data
        if pd.isna(md.iloc[i]) or pd.isna(cdir.iloc[i]) or pd.isna(nVolume.iloc[i]):
            continue
        # previous md for direction check
        md_prev = md.iloc[i - 1] if i > 0 else np.nan
        if pd.isna(md_prev):
            continue

        close_i = price.iloc[i]
        long_cond = (cdir.iloc[i] == 1) and (nVolume.iloc[i] >= hv_threshold) and (md.iloc[i] > md_prev) and (close_i > md.iloc[i])
        short_cond = (cdir.iloc[i] == -1) and (nVolume.iloc[i] >= hv_threshold) and (md.iloc[i] < md_prev) and (close_i < md.iloc[i])

        if long_cond:
            ts = int(df['time'].iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close_i),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_i),
                'raw_price_b': float(close_i)
            }
            entries.append(entry)
            trade_num += 1
        elif short_cond:
            ts = int(df['time'].iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close_i),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_i),
                'raw_price_b': float(close_i)
            }
            entries.append(entry)
            trade_num += 1

    return entries