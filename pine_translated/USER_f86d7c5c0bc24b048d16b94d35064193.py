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
    tradeDirection = "Both"
    atrLength = 14
    almaLength = 50
    offset = 0.85
    sigma = 6.0
    bb = 20
    input_retSince = 2
    input_retValid = 2
    rTon = True
    rTcc = False
    rThv = False

    def calc_alma(prices, length, offset_val, sigma_val):
        rng = np.arange(length)
        w = np.exp(-np.power(rng - offset_val * (length - 1), 2) / (2 * sigma_val * sigma_val))
        w = w / w.sum()
        result = np.convolve(prices, w, mode='valid')
        return pd.Series(np.concatenate([np.full(length - 1, np.nan), result]), index=prices.index)

    df['alma'] = calc_alma(df['close'], almaLength, offset, sigma)

    low_arr = df['low'].values
    high_arr = df['high'].values
    n = len(df)
    window_size = bb + 1

    pl = np.full(n, np.nan)
    ph = np.full(n, np.nan)
    for i in range(window_size, n):
        pl[i] = np.min(low_arr[i - window_size + 1:i + 1])
        ph[i] = np.max(high_arr[i - window_size + 1:i + 1])

    pl_series = pd.Series(pl).ffill()
    ph_series = pd.Series(ph).ffill()

    def get_pivot_val(pivot_arr, idx, bb_val):
        if idx - bb_val < 0:
            return np.nan
        return pivot_arr[idx - bb_val]

    pl_vals = np.array([get_pivot_val(pl, i, bb) for i in range(n)])
    ph_vals = np.array([get_pivot_val(ph, i, bb) for i in range(n)])

    sBot_vals = np.where(low_arr[bb + 1:] > low_arr[bb - 1:-1],
                         low_arr[bb - 1:-1], low_arr[bb + 1:]) if n > bb + 1 else np.full(n, np.nan)
    sTop_vals = np.where(high_arr[bb + 1:] > high_arr[bb - 1:-1],
                         high_arr[bb + 1:], high_arr[bb - 1:-1]) if n > bb + 1 else np.full(n, np.nan)
    rBot_vals = np.where(low_arr[bb + 1:] > low_arr[bb - 1:-1],
                         low_arr[bb + 1:], low_arr[bb - 1:-1]) if n > bb + 1 else np.full(n, np.nan)
    rTop_vals = np.where(high_arr[bb + 1:] > high_arr[bb - 1:-1],
                         high_arr[bb + 1:], high_arr[bb - 1:-1]) if n > bb + 1 else np.full(n, np.nan)

    sBot = pd.Series(np.concatenate([[np.nan] * (bb + 1), sBot_vals]), index=df.index)
    sTop = pd.Series(np.concatenate([[np.nan] * (bb + 1), sTop_vals]), index=df.index)
    rBot = pd.Series(np.concatenate([[np.nan] * (bb + 1), rBot_vals]), index=df.index)
    rTop = pd.Series(np.concatenate([[np.nan] * (bb + 1), rTop_vals]), index=df.index)

    co = pd.Series(np.nan, index=df.index)
    cu = pd.Series(np.nan, index=df.index)

    for i in range(1, n):
        if rTon:
            co.iloc[i] = df['close'].iloc[i] > rTop.iloc[i] and df['close'].iloc[i - 1] <= rTop.iloc[i - 1]
            cu.iloc[i] = df['close'].iloc[i] < sBot.iloc[i] and df['close'].iloc[i - 1] >= sBot.iloc[i - 1]
        elif rThv:
            co.iloc[i] = df['high'].iloc[i] > rTop.iloc[i] and df['high'].iloc[i - 1] <= rTop.iloc[i - 1]
            cu.iloc[i] = df['low'].iloc[i] < sBot.iloc[i] and df['low'].iloc[i - 1] >= sBot.iloc[i - 1]
        elif rTcc:
            co.iloc[i] = df['close'].iloc[i] > rTop.iloc[i] and df['close'].iloc[i - 1] <= rTop.iloc[i - 1]
            cu.iloc[i] = df['close'].iloc[i] < sBot.iloc[i] and df['close'].iloc[i - 1] >= sBot.iloc[i - 1]

    co = co.astype(bool)
    cu = cu.astype(bool)

    pl_change = (pl_series != pl_series.shift(1)) & pl_series.notna()
    ph_change = (ph_series != ph_series.shift(1)) & ph_series.notna()

    sBreak = None
    rBreak = None
    trade_num = 1
    entries = []

    def barssince(cond_arr, idx):
        if idx < 0:
            return -1
        for j in range(idx, -1, -1):
            if cond_arr[j]:
                return idx - j
        return -1

    for i in range(bb + 1, n):
        if pl_change.iloc[i]:
            if sBreak is None or (isinstance(sBreak, float) and np.isnan(sBreak)):
                sBreak = None
        if ph_change.iloc[i]:
            if rBreak is None or (isinstance(rBreak, float) and np.isnan(rBreak)):
                rBreak = None

        if co.iloc[i] and (sBreak is None or (isinstance(sBreak, float) and np.isnan(sBreak))):
            sBreak = True
        if cu.iloc[i] and (rBreak is None or (isinstance(rBreak, float) and np.isnan(rBreak))):
            rBreak = True

        s_bs = barssince(co.values, i) if sBreak else -1
        r_bs = barssince(cu.values, i) if rBreak else -1

        s1 = (s_bs > input_retSince) and (df['high'].iloc[i] >= sTop.iloc[i]) and (df['close'].iloc[i] <= sBot.iloc[i])
        s2 = (s_bs > input_retSince) and (df['high'].iloc[i] >= sTop.iloc[i]) and (df['close'].iloc[i] >= sBot.iloc[i]) and (df['close'].iloc[i] <= sTop.iloc[i])
        s3 = (s_bs > input_retSince) and (df['high'].iloc[i] >= sBot.iloc[i]) and (df['high'].iloc[i] <= sTop.iloc[i])
        s4 = (s_bs > input_retSince) and (df['high'].iloc[i] >= sBot.iloc[i]) and (df['high'].iloc[i] <= sTop.iloc[i]) and (df['close'].iloc[i] < sBot.iloc[i])

        r1 = (r_bs > input_retSince) and (df['low'].iloc[i] <= rBot.iloc[i]) and (df['close'].iloc[i] >= rTop.iloc[i])
        r2 = (r_bs > input_retSince) and (df['low'].iloc[i] <= rBot.iloc[i]) and (df['close'].iloc[i] <= rTop.iloc[i]) and (df['close'].iloc[i] >= rBot.iloc[i])
        r3 = (r_bs > input_retSince) and (df['low'].iloc[i] <= rTop.iloc[i]) and (df['low'].iloc[i] >= rBot.iloc[i])
        r4 = (r_bs > input_retSince) and (df['low'].iloc[i] <= rTop.iloc[i]) and (df['low'].iloc[i] >= rBot.iloc[i]) and (df['close'].iloc[i] > rTop.iloc[i])

        sRetValid = s_bs > 0 and s_bs <= input_retValid and (s1 or s2 or s3 or s4)
        rRetValid = r_bs > 0 and r_bs <= input_retValid and (r1 or r2 or r3 or r4)

        if tradeDirection in ["Long", "Both"] and sRetValid:
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        if tradeDirection in ["Short", "Both"] and rRetValid:
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries