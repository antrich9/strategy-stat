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
    # Parameters
    htf_enabled = True
    fvg_enabled = True
    fvg_min_size = 0.05
    fib_retracement_level = 0.71
    liquidity_lookback = 20

    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    time = df['time'].values

    # ========== Wilder ATR(14) ==========
    tr = np.maximum(high[1:] - low[1:], np.maximum(
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    ))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    if n > 0:
        atr[0] = np.mean(tr[1:15]) if n > 1 else tr[1]
        alpha = 1.0 / 14.0
        for i in range(1, n):
            if i < 14:
                atr[i] = np.nanmean(tr[1:i+1])
            else:
                atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]

    # ========== Swing High/Low (Pivot High/Low) ==========
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    for i in range(liquidity_lookback, n):
        is_high = True
        for j in range(i - liquidity_lookback, i + 1):
            if high[j] > high[i]:
                is_high = False
                break
        if is_high:
            swing_high[i] = high[i]
        is_low = True
        for j in range(i - liquidity_lookback, i + 1):
            if low[j] < low[i]:
                is_low = False
                break
        if is_low:
            swing_low[i] = low[i]

    # ========== Recent Swing Levels (var float) ==========
    recent_swing_high = np.full(n, np.nan)
    recent_swing_low = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(swing_high[i]):
            recent_swing_high[i] = swing_high[i]
        elif i > 0:
            recent_swing_high[i] = recent_swing_high[i-1]
        if not np.isnan(swing_low[i]):
            recent_swing_low[i] = swing_low[i]
        elif i > 0:
            recent_swing_low[i] = recent_swing_low[i-1]

    # ========== Liquidity Sweep Detection ==========
    liquidity_sweep_bull = np.full(n, False)
    liquidity_sweep_bear = np.full(n, False)
    for i in range(1, n):
        if not np.isnan(recent_swing_low[i]) and low[i] < recent_swing_low[i] and close[i] > recent_swing_low[i]:
            liquidity_sweep_bull[i] = True
        if not np.isnan(recent_swing_high[i]) and high[i] > recent_swing_high[i] and close[i] < recent_swing_high[i]:
            liquidity_sweep_bear[i] = True

    # ========== Break of Structure ==========
    bos_direction = [None] * n
    for i in range(1, n):
        if liquidity_sweep_bull[i] and close[i] > high[i-1]:
            bos_direction[i] = "bull"
        elif liquidity_sweep_bear[i] and close[i] < low[i-1]:
            bos_direction[i] = "bear"
        else:
            bos_direction[i] = bos_direction[i-1] if i > 0 else None

    # ========== Fair Value Gap Detection ==========
    bull_fvg = np.full(n, False)
    bear_fvg = np.full(n, False)
    for i in range(2, n):
        if low[i] > high[i-2] and (low[i] - high[i-2]) / close[i] > fvg_min_size / 100:
            bull_fvg[i] = True
        if high[i] < low[i-2] and (low[i-2] - high[i]) / close[i] > fvg_min_size / 100:
            bear_fvg[i] = True

    bull_fvg_present = np.full(n, False)
    bear_fvg_present = np.full(n, False)
    for i in range(n):
        bull_fvg_present[i] = not fvg_enabled or (i-1 >= 0 and bull_fvg[i-1]) or (i-2 >= 0 and bull_fvg[i-2]) or (i-3 >= 0 and bull_fvg[i-3])
        bear_fvg_present[i] = not fvg_enabled or (i-1 >= 0 and bear_fvg[i-1]) or (i-2 >= 0 and bear_fvg[i-2]) or (i-3 >= 0 and bear_fvg[i-3])

    # ========== Fibonacci Levels ==========
    fib_71_level = np.full(n, np.nan)
    for i in range(n):
        if bos_direction[i] == "bull" and not np.isnan(recent_swing_low[i]) and not np.isnan(recent_swing_high[i]):
            fib_high = recent_swing_high[i]
            fib_low = recent_swing_low[i]
            fib_71_level[i] = fib_high - (fib_high - fib_low) * fib_retracement_level
        elif bos_direction[i] == "bear" and not np.isnan(recent_swing_low[i]) and not np.isnan(recent_swing_high[i]):
            fib_high = recent_swing_high[i]
            fib_low = recent_swing_low[i]
            fib_71_level[i] = fib_low + (fib_high - fib_low) * fib_retracement_level

    # ========== Entry Conditions ==========
    bullish_entry = np.full(n, False)
    bearish_entry = np.full(n, False)
    for i in range(n):
        if np.isnan(atr[i]):
            continue
        bull_htf_ok = not htf_enabled
        bull_liquidity_sweep = liquidity_sweep_bull[i]
        bull_bos = bos_direction[i] == "bull"
        bull_at_fib_level = not np.isnan(fib_71_level[i]) and close[i] <= fib_71_level[i] and close[i] >= fib_71_level[i] * 0.99
        if bull_htf_ok and bull_liquidity_sweep and bull_bos and bull_fvg_present[i] and bull_at_fib_level:
            bullish_entry[i] = True

        bear_htf_ok = not htf_enabled
        bear_liquidity_sweep = liquidity_sweep_bear[i]
        bear_bos = bos_direction[i] == "bear"
        bear_at_fib_level = not np.isnan(fib_71_level[i]) and close[i] >= fib_71_level[i] and close[i] <= fib_71_level[i] * 1.01
        if bear_htf_ok and bear_liquidity_sweep and bear_bos and bear_fvg_present[i] and bear_at_fib_level:
            bearish_entry[i] = True

    # ========== Generate Entries ==========
    entries = []
    trade_num = 1
    for i in range(n):
        if bullish_entry[i]:
            entry_price = float(close[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(time[i]),
                'entry_time': datetime.fromtimestamp(time[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        if bearish_entry[i]:
            entry_price = float(close[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(time[i]),
                'entry_time': datetime.fromtimestamp(time[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries