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
    bb = 20
    input_retSince = 2
    input_retValid = 2
    rTon = True
    rTcc = False
    rThv = False

    close = df['close']
    high = df['high']
    low = df['low']

    # Pivot calculation
    pl = pd.Series(index=df.index, dtype=float)
    ph = pd.Series(index=df.index, dtype=float)

    for i in range(bb, len(df)):
        pl_vals = []
        ph_vals = []
        for j in range(1, bb + 1):
            if i - j >= 0:
                pl_vals.append(df['low'].iloc[i - j])
            if i - j >= 0:
                ph_vals.append(df['high'].iloc[i - j])
        pl.iloc[i] = min(pl_vals) if pl_vals else np.nan
        ph.iloc[i] = max(ph_vals) if ph_vals else np.nan

    pl = pl.replace(0, np.nan).ffill().bfill()
    ph = ph.replace(0, np.nan).ffill().bfill()

    # Box heights
    s_yLoc = pd.Series(index=df.index, dtype=float)
    r_yLoc = pd.Series(index=df.index, dtype=float)

    for i in range(bb + 1, len(df)):
        s_yLoc.iloc[i] = df['low'].iloc[bb - 1] if df['low'].iloc[bb + 1] > df['low'].iloc[bb - 1] else df['low'].iloc[bb + 1]
        r_yLoc.iloc[i] = df['high'].iloc[bb + 1] if df['high'].iloc[bb + 1] > df['high'].iloc[bb - 1] else df['high'].iloc[bb - 1]

    s_yLoc = s_yLoc.fillna(method='bfill')
    r_yLoc = r_yLoc.fillna(method='bfill')

    # ATR calculation (Wilder ATR)
    atr = pd.Series(index=df.index, dtype=float)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr.iloc[0] = tr.iloc[0]
    for i in range(1, len(df)):
        atr.iloc[i] = (atr.iloc[i - 1] * (atr - 1) + tr.iloc[i]) / atr

    atr = atr.ewm(alpha=1/14, adjust=False).mean()

    # Box top/bottom
    sTop = pd.Series(index=df.index, dtype=float)
    rTop = pd.Series(index=df.index, dtype=float)
    sBot = pd.Series(index=df.index, dtype=float)
    rBot = pd.Series(index=df.index, dtype=float)

    for i in range(len(df)):
        if pd.notna(pl.iloc[i]) and i >= bb:
            sTop.iloc[i] = pl.iloc[i]
            sBot.iloc[i] = s_yLoc.iloc[i] if pd.notna(s_yLoc.iloc[i]) else pl.iloc[i]
        if pd.notna(ph.iloc[i]) and i >= bb:
            rBot.iloc[i] = ph.iloc[i]
            rTop.iloc[i] = r_yLoc.iloc[i] if pd.notna(r_yLoc.iloc[i]) else ph.iloc[i]

    # Breakout detection
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)

    cu = pd.Series(False, index=df.index)
    co = pd.Series(False, index=df.index)

    for i in range(1, len(df)):
        if rTcc:
            cu.iloc[i] = close.iloc[i] < sBot.iloc[i] and close.iloc[i - 1] >= sBot.iloc[i - 1] if pd.notna(sBot.iloc[i]) and pd.notna(sBot.iloc[i - 1]) else False
            co.iloc[i] = close.iloc[i] > rTop.iloc[i] and close.iloc[i - 1] <= rTop.iloc[i - 1] if pd.notna(rTop.iloc[i]) and pd.notna(rTop.iloc[i - 1]) else False
        elif rThv:
            cu.iloc[i] = low.iloc[i] < sBot.iloc[i] and low.iloc[i - 1] >= sBot.iloc[i - 1] if pd.notna(sBot.iloc[i]) and pd.notna(sBot.iloc[i - 1]) else False
            co.iloc[i] = high.iloc[i] > rTop.iloc[i] and high.iloc[i - 1] <= rTop.iloc[i - 1] if pd.notna(rTop.iloc[i]) and pd.notna(rTop.iloc[i - 1]) else False
        else:
            cu.iloc[i] = close.iloc[i] < sBot.iloc[i] and close.iloc[i - 1] >= sBot.iloc[i - 1] if pd.notna(sBot.iloc[i]) and pd.notna(sBot.iloc[i - 1]) else False
            co.iloc[i] = close.iloc[i] > rTop.iloc[i] and close.iloc[i - 1] <= rTop.iloc[i - 1] if pd.notna(rTop.iloc[i]) and pd.notna(rTop.iloc[i - 1]) else False

        if cu.iloc[i] and not sBreak.iloc[i - 1]:
            sBreak.iloc[i] = True
        if co.iloc[i] and not rBreak.iloc[i - 1]:
            rBreak.iloc[i] = True

    # Retest conditions
    s1 = pd.Series(False, index=df.index)
    s2 = pd.Series(False, index=df.index)
    s3 = pd.Series(False, index=df.index)
    s4 = pd.Series(False, index=df.index)
    r1 = pd.Series(False, index=df.index)
    r2 = pd.Series(False, index=df.index)
    r3 = pd.Series(False, index=df.index)
    r4 = pd.Series(False, index=df.index)

    for i in range(bb + input_retSince + 1, len(df)):
        if sBreak.iloc[i - 1]:
            bars_since_break = 0
            for j in range(i - 1, -1, -1):
                if sBreak.iloc[j]:
                    bars_since_break = i - j - 1
                    break
            if bars_since_break > input_retSince:
                if pd.notna(sTop.iloc[i]) and pd.notna(sBot.iloc[i]):
                    s1.iloc[i] = high.iloc[i] >= sTop.iloc[i] and close.iloc[i] <= sBot.iloc[i]
                    s2.iloc[i] = high.iloc[i] >= sTop.iloc[i] and close.iloc[i] >= sBot.iloc[i] and close.iloc[i] <= sTop.iloc[i]
                    s3.iloc[i] = high.iloc[i] >= sBot.iloc[i] and high.iloc[i] <= sTop.iloc[i]
                    s4.iloc[i] = high.iloc[i] >= sBot.iloc[i] and high.iloc[i] <= sTop.iloc[i] and close.iloc[i] < sBot.iloc[i]
        if rBreak.iloc[i - 1]:
            bars_since_break = 0
            for j in range(i - 1, -1, -1):
                if rBreak.iloc[j]:
                    bars_since_break = i - j - 1
                    break
            if bars_since_break > input_retSince:
                if pd.notna(rTop.iloc[i]) and pd.notna(rBot.iloc[i]):
                    r1.iloc[i] = low.iloc[i] <= rBot.iloc[i] and close.iloc[i] >= rTop.iloc[i]
                    r2.iloc[i] = low.iloc[i] <= rBot.iloc[i] and close.iloc[i] <= rTop.iloc[i] and close.iloc[i] >= rBot.iloc[i]
                    r3.iloc[i] = low.iloc[i] <= rTop.iloc[i] and low.iloc[i] >= rBot.iloc[i]
                    r4.iloc[i] = low.iloc[i] <= rTop.iloc[i] and low.iloc[i] >= rBot.iloc[i] and close.iloc[i] > rTop.iloc[i]

    sRetValid = pd.Series(False, index=df.index)
    rRetValid = pd.Series(False, index=df.index)

    retOccurred_s = False
    retOccurred_r = False

    for i in range(1, len(df)):
        sRetActive = s1.iloc[i] or s2.iloc[i] or s3.iloc[i] or s4.iloc[i]
        rRetActive = r1.iloc[i] or r2.iloc[i] or r3.iloc[i] or r4.iloc[i]

        if sRetActive and not sRetActive:
            retOccurred_s = False

        if rRetActive and not rRetActive:
            retOccurred_r = False

        if sRetActive:
            retOccurred_s = False

        if rRetActive:
            retOccurred_r = False

        retValid_s = False
        retValid_r = False

        if sRetActive:
            bars_since = 0
            for j in range(i - 1, -1, -1):
                sRetPrev = s1.iloc[j] or s2.iloc[j] or s3.iloc[j] or s4.iloc[j]
                if sRetPrev:
                    bars_since = i - j
                    break
            if bars_since > 0 and bars_since <= input_retValid:
                if rTon:
                    retValid_s = close.iloc[i] <= pl.iloc[i] if pd.notna(pl.iloc[i]) else False
                elif rThv:
                    retValid_s = low.iloc[i] <= pl.iloc[i] if pd.notna(pl.iloc[i]) else False
                elif rTcc:
                    retValid_s = close.iloc[i] <= pl.iloc[i] if pd.notna(pl.iloc[i]) else False
                if retValid_s and not retOccurred_s:
                    sRetValid.iloc[i] = True
                    retOccurred_s = True

        if rRetActive:
            bars_since = 0
            for j in range(i - 1, -1, -1):
                rRetPrev = r1.iloc[j] or r2.iloc[j] or r3.iloc[j] or r4.iloc[j]
                if rRetPrev:
                    bars_since = i - j
                    break
            if bars_since > 0 and bars_since <= input_retValid:
                if rTon:
                    retValid_r = close.iloc[i] >= ph.iloc[i] if pd.notna(ph.iloc[i]) else False
                elif rThv:
                    retValid_r = high.iloc[i] >= ph.iloc[i] if pd.notna(ph.iloc[i]) else False
                elif rTcc:
                    retValid_r = close.iloc[i] >= ph.iloc[i] if pd.notna(ph.iloc[i]) else False
                if retValid_r and not retOccurred_r:
                    rRetValid.iloc[i] = True
                    retOccurred_r = True

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(close.iloc[i]):
            continue

        if sRetValid.iloc[i] and sBreak.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

        if rRetValid.iloc[i] and rBreak.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

    return entries