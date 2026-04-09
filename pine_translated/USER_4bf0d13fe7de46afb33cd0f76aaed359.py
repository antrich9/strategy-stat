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
    # Parameters from the strategy
    atrLength = 14
    atrMultiplier = 1.5
    takeProfitRatio = 1.5
    tradeDirection = "Both"
    dpoLength = 20
    dpoDisplace = int(np.ceil(dpoLength / 2)) + 1
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    input_breakout = True
    input_retest = True
    input_repType = 'On'
    bb = input_lookback

    rTon = input_repType == 'On'
    rTcc = input_repType == 'Off: Candle Confirmation'
    rThv = input_repType == 'Off: High & Low'

    n = len(df)
    close = df['close']
    high = df['high']
    low = df['low']

    # Calculate pivot points (pl for support, ph for resistance)
    pl = pd.Series(np.nan, index=df.index)
    ph = pd.Series(np.nan, index=df.index)

    for i in range(bb, n):
        # pivotlow: lowest low over bb bars ending at i-bb
        start_idx = i - bb
        end_idx = i
        window = low.iloc[start_idx:end_idx + 1]
        min_idx = window.idxmin()
        if min_idx == i - bb:
            pl.iloc[i] = low.iloc[i - bb]

        # pivothigh: highest high over bb bars ending at i-bb
        window_h = high.iloc[start_idx:end_idx + 1]
        max_idx = window_h.idxmax()
        if max_idx == i - bb:
            ph.iloc[i] = high.iloc[i - bb]

    # Forward fill NaN with last valid value (fixnan behavior)
    pl = pl.ffill()
    ph = ph.ffill()

    # Calculate box top and bottom for support (sBox) and resistance (rBox)
    sTop = pd.Series(np.nan, index=df.index)
    sBot = pd.Series(np.nan, index=df.index)
    rTop = pd.Series(np.nan, index=df.index)
    rBot = pd.Series(np.nan, index=df.index)

    for i in range(bb, n):
        if not pd.isna(pl.iloc[i]):
            s_yLoc = low.iloc[bb + 1] if bb + 1 <= i else low.iloc[1]
            if i - 1 >= 0:
                s_yLoc = low.iloc[i - 1] if low.iloc[bb + 1] > low.iloc[i - 1] else low.iloc[bb + 1]
            sTop.iloc[i] = pl.iloc[i]
            sBot.iloc[i] = s_yLoc if s_yLoc < pl.iloc[i] else pl.iloc[i]

        if not pd.isna(ph.iloc[i]):
            r_yLoc = high.iloc[bb + 1] if bb + 1 <= i else high.iloc[1]
            if i - 1 >= 0:
                r_yLoc = high.iloc[i - 1] if high.iloc[bb + 1] > high.iloc[i - 1] else high.iloc[bb + 1]
            rBot.iloc[i] = ph.iloc[i]
            rTop.iloc[i] = r_yLoc if r_yLoc > ph.iloc[i] else ph.iloc[i]

    # Wilder's ATR calculation
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = pd.Series(np.nan, index=df.index)
    alpha = 1.0 / atrLength
    atr_vals = []
    first_valid = tr.first_valid_index()
    if first_valid is not None:
        atr_vals = [tr.iloc[first_valid]]
        for i in range(first_valid + 1, n):
            atr_val = alpha * tr.iloc[i] + (1 - alpha) * atr_vals[-1]
            atr_vals.append(atr_val)
        for i, idx in enumerate(df.index[first_valid:first_valid + len(atr_vals)]):
            atr.iloc[idx] = atr_vals[i]

    # Calculate stop loss and take profit levels
    stopLossLong = close - (atr * atrMultiplier)
    stopLossShort = close + (atr * atrMultiplier)
    takeProfitLong = close + ((atr * atrMultiplier) * takeProfitRatio)
    takeProfitShort = close - ((atr * atrMultiplier) * takeProfitRatio)

    # Breakout detection
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)
    cu_series = pd.Series(False, index=df.index)
    co_series = pd.Series(False, index=df.index)

    for i in range(bb + 1, n):
        if i >= 1:
            if rTon:
                cu_val = close.iloc[i] < sBot.iloc[i] and close.iloc[i - 1] >= sBot.iloc[i]
                co_val = close.iloc[i] > rTop.iloc[i] and close.iloc[i - 1] <= rTop.iloc[i]
            elif rThv:
                cu_val = low.iloc[i] < sBot.iloc[i] and low.iloc[i - 1] >= sBot.iloc[i]
                co_val = high.iloc[i] > rTop.iloc[i] and high.iloc[i - 1] <= rTop.iloc[i]
            else:
                cu_val = (close.iloc[i] < sBot.iloc[i] and close.iloc[i - 1] >= sBot.iloc[i]) if i > 0 else False
                co_val = (close.iloc[i] > rTop.iloc[i] and close.iloc[i - 1] <= rTop.iloc[i]) if i > 0 else False

            cu_series.iloc[i] = cu_val
            co_series.iloc[i] = co_val

    # Update sBreak and rBreak flags
    pl_change = pl != pl.shift(1)
    ph_change = ph != ph.shift(1)

    for i in range(bb, n):
        if cu_series.iloc[i] and not sBreak.iloc[i - 1] if i > 0 else cu_series.iloc[i]:
            sBreak.iloc[i] = True
        else:
            sBreak.iloc[i] = sBreak.iloc[i - 1] if i > 0 else False

        if co_series.iloc[i] and not rBreak.iloc[i - 1] if i > 0 else co_series.iloc[i]:
            rBreak.iloc[i] = True
        else:
            rBreak.iloc[i] = rBreak.iloc[i - 1] if i > 0 else False

        if pl_change.iloc[i]:
            if pd.isna(pl.iloc[i]) or (i > 0 and pd.isna(pl.iloc[i - 1])):
                sBreak.iloc[i] = False
        if ph_change.iloc[i]:
            if pd.isna(ph.iloc[i]) or (i > 0 and pd.isna(ph.iloc[i - 1])):
                rBreak.iloc[i] = False

    # Retest conditions for support (s1, s2, s3, s4)
    s1 = pd.Series(False, index=df.index)
    s2 = pd.Series(False, index=df.index)
    s3 = pd.Series(False, index=df.index)
    s4 = pd.Series(False, index=df.index)

    # Retest conditions for resistance (r1, r2, r3, r4)
    r1 = pd.Series(False, index=df.index)
    r2 = pd.Series(False, index=df.index)
    r3 = pd.Series(False, index=df.index)
    r4 = pd.Series(False, index=df.index)

    for i in range(bb + 1, n):
        if sBreak.iloc[i]:
            bars_since_break = i - (i - 1) if not sBreak.iloc[i - 1] else 0
            for j in range(i - 1, -1, -1):
                if sBreak.iloc[j]:
                    bars_since_break = i - j
                    break
            if bars_since_break > input_retSince:
                if high.iloc[i] >= sTop.iloc[i] and close.iloc[i] <= sBot.iloc[i]:
                    s1.iloc[i] = True
                if high.iloc[i] >= sTop.iloc[i] and close.iloc[i] >= sBot.iloc[i] and close.iloc[i] <= sTop.iloc[i]:
                    s2.iloc[i] = True
                if high.iloc[i] >= sBot.iloc[i] and high.iloc[i] <= sTop.iloc[i]:
                    s3.iloc[i] = True
                if high.iloc[i] >= sBot.iloc[i] and high.iloc[i] <= sTop.iloc[i] and close.iloc[i] < sBot.iloc[i]:
                    s4.iloc[i] = True

        if rBreak.iloc[i]:
            bars_since_break = i - (i - 1) if not rBreak.iloc[i - 1] else 0
            for j in range(i - 1, -1, -1):
                if rBreak.iloc[j]:
                    bars_since_break = i - j
                    break
            if bars_since_break > input_retSince:
                if low.iloc[i] <= rBot.iloc[i] and close.iloc[i] >= rTop.iloc[i]:
                    r1.iloc[i] = True
                if low.iloc[i] <= rBot.iloc[i] and close.iloc[i] <= rTop.iloc[i] and close.iloc[i] >= rBot.iloc[i]:
                    r2.iloc[i] = True
                if low.iloc[i] <= rTop.iloc[i] and low.iloc[i] >= rBot.iloc[i]:
                    r3.iloc[i] = True
                if low.iloc[i] <= rTop.iloc[i] and low.iloc[i] >= rBot.iloc[i] and close.iloc[i] > rTop.iloc[i]:
                    r4.iloc[i] = True

    # Retest event detection (simplified version)
    sRetValid = pd.Series(False, index=df.index)
    rRetValid = pd.Series(False, index=df.index)

    retOccurred_s = False
    retOccurred_r = False

    for i in range(bb + 1, n):
        s_ret_active = s1.iloc[i] or s2.iloc[i] or s3.iloc[i] or s4.iloc[i]
        r_ret_active = r1.iloc[i] or r2.iloc[i] or r3.iloc[i] or r4.iloc[i]

        if input_retest:
            if s_ret_active and (i == 0 or not s_ret_active):
                retOccurred_s = False
            if r_ret_active and (i == 0 or not r_ret_active):
                retOccurred_r = False

            # Calculate bars since retest event
            bars_since_s_ret = 0
            bars_since_r_ret = 0
            for j in range(i - 1, -1, -1):
                if s_ret_active and (j == 0 or not (s1.iloc[j] or s2.iloc[j] or s3.iloc[j] or s4.iloc[j])):
                    bars_since_s_ret = i - j
                    break
            for j in range(i - 1, -1, -1):
                if r_ret_active and (j == 0 or not (r1.iloc[j] or r2.iloc[j] or r3.iloc[j] or r4.iloc[j])):
                    bars_since_r_ret = i - j
                    break

            if rTon:
                s_ret_cond = close.iloc[i] <= sBot.iloc[i] if not pd.isna(sBot.iloc[i]) else False
                r_ret_cond = close.iloc[i] >= rTop.iloc[i] if not pd.isna(rTop.iloc[i]) else False
            elif rThv:
                s_ret_cond = low.iloc[i] <= sBot.iloc[i] if not pd.isna(sBot.iloc[i]) else False
                r_ret_cond = high.iloc[i] >= rTop.iloc[i] if not pd.isna(rTop.iloc[i]) else False
            else:
                s_ret_cond = (close.iloc[i] <= sBot.iloc[i] and (i == 0 or close.iloc[i - 1] > sBot.iloc[i - 1])) if not pd.isna(sBot.iloc[i]) else False
                r_ret_cond = (close.iloc[i] >= rTop.iloc[i] and (i == 0 or close.iloc[i - 1] < rTop.iloc[i - 1])) if not pd.isna(rTop.iloc[i]) else False

            if bars_since_s_ret > 0 and bars_since_s_ret <= input_retValid and s_ret_cond and not retOccurred_s:
                sRetValid.iloc[i] = True
                retOccurred_s = True

            if bars_since_r_ret > 0 and bars_since_r_ret <= input_retValid and r_ret_cond and not retOccurred_r:
                rRetValid.iloc[i] = True
                retOccurred_r = True

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(n):
        if pd.isna(atr.iloc[i]):
            continue

        if input_retest:
            if tradeDirection in ["Long", "Both"] and sRetValid.iloc[i]:
                entry_ts = int(df['time'].iloc[i])
                entry_price = float(close.iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1

            if tradeDirection in ["Short", "Both"] and rRetValid.iloc[i]:
                entry_ts = int(df['time'].iloc[i])
                entry_price = float(close.iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1

    return entries