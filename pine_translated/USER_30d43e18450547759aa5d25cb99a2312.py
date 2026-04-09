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
    o = df['open'].copy()
    h = df['high'].copy()
    l = df['low'].copy()
    c = df['close'].copy()
    ts = df['time'].copy()

    atrLength = 14
    atrMultiplier = 1.5
    maMethod = 'EMA'
    maLength1 = 6
    maLength2 = 4
    tradeDirection = 'Both'
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    rTon = True
    rTcc = False
    rThv = False

    haClose = (o + h + l + c) / 4

    haOpen = np.zeros(len(df))
    haOpen[0] = (o.iloc[0] + c.iloc[0]) / 2
    for i in range(1, len(df)):
        haOpen[i] = haOpen[i-1] + haClose.iloc[i-1]
        haOpen[i] /= 2

    haHigh = np.maximum(h.values, np.maximum(haOpen, haClose.values))
    haLow = np.minimum(l.values, np.minimum(haOpen, haClose.values))

    if maMethod == 'EMA':
        smoothHAOpen = pd.Series(haOpen).ewm(span=maLength1, adjust=False).mean()
        smoothHAClose = pd.Series(haClose).ewm(span=maLength2, adjust=False).mean()
    else:
        smoothHAOpen = pd.Series(haOpen).rolling(maLength1).mean()
        smoothHAClose = pd.Series(haClose).rolling(maLength2).mean()

    bb = input_lookback

    def wilders_atr(high, low, close, length):
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1).values)
        tr3 = np.abs(low - close.shift(1).values)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros(len(high))
        atr[length-1] = tr[:length].mean()
        alpha = 1 / length
        for i in range(length, len(high)):
            atr[i] = atr[i-1] * (1 - alpha) + tr[i] * alpha
        return pd.Series(atr, index=high.index)

    atr = wilders_atr(h, l, c, atrLength)

    pl = pd.Series(np.nan, index=df.index)
    ph = pd.Series(np.nan, index=df.index)
    for i in range(bb, len(df) - bb):
        pl_vals = l.iloc[i-bb:i+bb+1]
        ph_vals = h.iloc[i-bb:i+bb+1]
        if len(pl_vals) > 0:
            pl.iloc[i] = pl_vals.min()
        if len(ph_vals) > 0:
            ph.iloc[i] = ph_vals.max()
    pl = pl.fillna(method='ffill')
    ph = ph.fillna(method='ffill')

    s_yLoc = np.where(l.iloc[bb + 1] > l.iloc[bb - 1], l.iloc[bb - 1], l.iloc[bb + 1])
    r_yLoc = np.where(h.iloc[bb + 1] > h.iloc[bb - 1], h.iloc[bb + 1], h.iloc[bb - 1])

    sBox_top = pd.Series(np.nan, index=df.index)
    sBox_bot = pd.Series(np.nan, index=df.index)
    rBox_top = pd.Series(np.nan, index=df.index)
    rBox_bot = pd.Series(np.nan, index=df.index)

    for i in range(bb, len(df)):
        if pd.notna(pl.iloc[i]) and (i == 0 or pl.iloc[i] != pl.iloc[i-1]):
            sBox_top.iloc[i] = s_yLoc if i == bb else sBox_top.iloc[i-1]
            sBox_bot.iloc[i] = pl.iloc[i]
        elif i > bb:
            sBox_top.iloc[i] = sBox_top.iloc[i-1] if pd.notna(sBox_top.iloc[i-1]) else np.nan
            sBox_bot.iloc[i] = sBox_bot.iloc[i-1] if pd.notna(sBox_bot.iloc[i-1]) else np.nan

    for i in range(bb, len(df)):
        if pd.notna(ph.iloc[i]) and (i == 0 or ph.iloc[i] != ph.iloc[i-1]):
            rBox_top.iloc[i] = ph.iloc[i]
            rBox_bot.iloc[i] = r_yLoc if i == bb else rBox_bot.iloc[i-1]
        elif i > bb:
            rBox_top.iloc[i] = rBox_top.iloc[i-1] if pd.notna(rBox_top.iloc[i-1]) else np.nan
            rBox_bot.iloc[i] = rBox_bot.iloc[i-1] if pd.notna(rBox_bot.iloc[i-1]) else np.nan

    def repaint_func(c1, c2, c3):
        if rTon:
            return c1
        elif rThv:
            return c2
        elif rTcc:
            return c3
        return c1

    cu = pd.Series(False, index=df.index)
    co = pd.Series(False, index=df.index)

    for i in range(1, len(df)):
        sBot_val = sBox_bot.iloc[i]
        rTop_val = rBox_top.iloc[i]
        close_val = c.iloc[i]
        low_val = l.iloc[i]
        high_val = h.iloc[i]

        if pd.notna(sBot_val):
            crossunder_no_conf = close_val < sBot_val and c.iloc[i-1] >= sBot_val
            crossunder_with_conf = close_val < sBot_val and c.iloc[i-1] >= sBot_val
            cu.iloc[i] = repaint_func(crossunder_no_conf, low_val < sBot_val and l.iloc[i-1] >= sBot_val, crossunder_with_conf)

        if pd.notna(rTop_val):
            crossover_no_conf = close_val > rTop_val and c.iloc[i-1] <= rTop_val
            crossover_with_conf = close_val > rTop_val and c.iloc[i-1] <= rTop_val
            co.iloc[i] = repaint_func(crossover_no_conf, high_val > rTop_val and h.iloc[i-1] <= rTop_val, crossover_with_conf)

    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)

    for i in range(1, len(df)):
        if pd.notna(pl.iloc[i]) and pd.notna(pl.iloc[i-1]) and pl.iloc[i] != pl.iloc[i-1]:
            if pd.isna(sBreak.iloc[i]):
                sBreak.iloc[i] = False
        else:
            sBreak.iloc[i] = sBreak.iloc[i-1] if pd.notna(sBreak.iloc[i-1]) else False

    for i in range(1, len(df)):
        if pd.notna(ph.iloc[i]) and pd.notna(ph.iloc[i-1]) and ph.iloc[i] != ph.iloc[i-1]:
            if pd.isna(rBreak.iloc[i]):
                rBreak.iloc[i] = False
        else:
            rBreak.iloc[i] = rBreak.iloc[i-1] if pd.notna(rBreak.iloc[i-1]) else False

    for i in range(bb, len(df)):
        if cu.iloc[i] and pd.isna(sBreak.iloc[i]):
            sBreak.iloc[i] = True
        if i > 0 and pd.notna(pl.iloc[i]) and pd.notna(pl.iloc[i-1]) and pl.iloc[i] != pl.iloc[i-1]:
            if pd.isna(sBreak.iloc[i]):
                sBreak.iloc[i] = False

    for i in range(bb, len(df)):
        if co.iloc[i] and pd.isna(rBreak.iloc[i]):
            rBreak.iloc[i] = True
        if i > 0 and pd.notna(ph.iloc[i]) and pd.notna(ph.iloc[i-1]) and ph.iloc[i] != ph.iloc[i-1]:
            if pd.isna(rBreak.iloc[i]):
                rBreak.iloc[i] = False

    sTop = sBox_top.copy()
    sBot = sBox_bot.copy()
    rTop = rBox_top.copy()
    rBot = rBox_bot.copy()

    sTop = sTop.fillna(method='ffill')
    sBot = sBot.fillna(method='ffill')
    rTop = rTop.fillna(method='ffill')
    rBot = rBot.fillna(method='ffill')

    def retestCondition(breakout, condition):
        bars_since_breakout = pd.Series(0, index=df.index)
        in_breakout = False
        for i in range(len(df)):
            if breakout.iloc[i]:
                in_breakout = True
                bars_since_breakout.iloc[i] = 0
            elif in_breakout:
                bars_since_breakout.iloc[i] = bars_since_breakout.iloc[i-1] + 1
            if condition.iloc[i] and bars_since_breakout.iloc[i] > input_retSince:
                return True
        return False

    s1 = pd.Series(False, index=df.index)
    s2 = pd.Series(False, index=df.index)
    s3 = pd.Series(False, index=df.index)
    s4 = pd.Series(False, index=df.index)

    for i in range(bb, len(df)):
        if sBreak.iloc[i]:
            sTop_val = sTop.iloc[i]
            sBot_val = sBot.iloc[i]
            high_val = h.iloc[i]
            close_val = c.iloc[i]
            if pd.notna(sTop_val) and pd.notna(sBot_val):
                if high_val >= sTop_val and close_val <= sBot_val:
                    s1.iloc[i] = True
                if high_val >= sTop_val and close_val >= sBot_val and close_val <= sTop_val:
                    s2.iloc[i] = True
                if high_val >= sBot_val and high_val <= sTop_val:
                    s3.iloc[i] = True
                if high_val >= sBot_val and high_val <= sTop_val and close_val < sBot_val:
                    s4.iloc[i] = True

    r1 = pd.Series(False, index=df.index)
    r2 = pd.Series(False, index=df.index)
    r3 = pd.Series(False, index=df.index)
    r4 = pd.Series(False, index=df.index)

    for i in range(bb, len(df)):
        if rBreak.iloc[i]:
            rTop_val = rTop.iloc[i]
            rBot_val = rBot.iloc[i]
            low_val = l.iloc[i]
            close_val = c.iloc[i]
            if pd.notna(rTop_val) and pd.notna(rBot_val):
                if low_val <= rBot_val and close_val >= rTop_val:
                    r1.iloc[i] = True
                if low_val <= rBot_val and close_val <= rTop_val and close_val >= rBot_val:
                    r2.iloc[i] = True
                if low_val <= rTop_val and low_val >= rBot_val:
                    r3.iloc[i] = True
                if low_val <= rTop_val and low_val >= rBot_val and close_val > rTop_val:
                    r4.iloc[i] = True

    def retest_event(c1, c2, c3, c4, y1, y2, col, style, pType, breakout_series):
        retValid = pd.Series(False, index=df.index)
        retOccurred = pd.Series(False, index=df.index)
        retEvent = pd.Series(False, index=df.index)
        retValue = pd.Series(np.nan, index=df.index)
        retSince = pd.Series(0, index=df.index)

        for i in range(bb, len(df)):
            c1_val = c1.iloc[i] if i < len(c1) else False
            c2_val = c2.iloc[i] if i < len(c2) else False
            c3_val = c3.iloc[i] if i < len(c3) else False
            c4_val = c4.iloc[i] if i < len(c4) else False

            retActive = c1_val or c2_val or c3_val or c4_val

            if i > 0:
                if retActive and not retActive:
                    retEvent.iloc[i] = True
            else:
                if retActive:
                    retEvent.iloc[i] = True

            if retEvent.iloc[i]:
                retOccurred.iloc[i] = False
                retValue.iloc[i] = y1.iloc[i]
                retSince.iloc[i] = 0

            if i > 0:
                if not pd.isna(retValue.iloc[i-1]):
                    retValue.iloc[i] = retValue.iloc[i-1]
                if retSince.iloc[i-1] >= 0:
                    retSince.iloc[i] = retSince.iloc[i-1] + 1

            if retActive:
                retEvent.iloc[i] = True

            if retEvent.iloc[i]:
                retValue.iloc[i] = y1.iloc[i]

            if retEvent.iloc[i]:
                if pType == 'ph':
                    cur_y2 = y2.iloc[i] if i < len(y2) else np.nan
                    prev_y2_vals = []
                    for j in range(i):
                        if retEvent.iloc[j]:
                            prev_y2_vals.append(y2.iloc[j] if j < len(y2) else np.nan)
                    if len(prev_y2_vals) > 0:
                        prev_y2_0 = prev_y2_vals[-1]
                        if pd.notna(cur_y2) and pd.notna(prev_y2_0):
                            if cur_y2 < prev_y2_0:
                                retEvent.iloc[i] = retActive
                else:
                    cur_y2 = y2.iloc[i] if i < len(y2) else np.nan
                    prev_y2_vals = []
                    for j in range(i):
                        if retEvent.iloc[j]:
                            prev_y2_vals.append(y2.iloc[j] if j < len(y2) else np.nan)
                    if len(prev_y2_vals) > 0:
                        prev_y2_0 = prev_y2_vals[-1]
                        if pd.notna(cur_y2) and pd.notna(prev_y2_0):
                            if cur_y2 > prev_y2_0:
                                retEvent.iloc[i] = retActive

            if retEvent.iloc[i]:
                retValue.iloc[i] = y1.iloc[i]

            bars_since = 0
            for j in range(i-1, -1, -1):
                if retEvent.iloc[j]:
                    break
                bars_since += 1

            if bars_since > 0 and bars_since <= input_retValid:
                if pType == 'ph':
                    cond = c.iloc[i] >= retValue.iloc[i] if i < len(c) else False
                else:
                    cond = c.iloc[i] <= retValue.iloc[i] if i < len(c) else False

                if cond and not retOccurred.iloc[i]:
                    retValid.iloc[i] = True
                    retOccurred.iloc[i] = True

        return retValid, retEvent

    sRetValid = pd.Series(False, index=df.index)
    rRetValid = pd.Series(False, index=df.index)

    sRetValid, _ = retest_event(s1, s2, s3, s4, h, l, None, None, 'pl', sBreak)
    rRetValid, _ = retest_event(r1, r2, r3, r4, l, h, None, None, 'ph', rBreak)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i < bb or pd.isna(sTop.iloc[i]) or pd.isna(sBot.iloc[i]):
            continue

        long_entry = sRetValid.iloc[i] if i < len(sRetValid) else False
        short_entry = rRetValid.iloc[i] if i < len(rRetValid) else False

        if tradeDirection == 'Long' or tradeDirection == 'Both':
            if long_entry:
                entry_price = c.iloc[i]
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(ts.iloc[i]),
                    'entry_time': datetime.fromtimestamp(ts.iloc[i] / 1000, tz=timezone.utc).isoformat() if ts.iloc[i] > 1e12 else datetime.fromtimestamp(ts.iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(entry_price),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(entry_price),
                    'raw_price_b': float(entry_price)
                })
                trade_num += 1

        if tradeDirection == 'Short' or tradeDirection == 'Both':
            if short_entry:
                entry_price = c.iloc[i]
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(ts.iloc[i]),
                    'entry_time': datetime.fromtimestamp(ts.iloc[i] / 1000, tz=timezone.utc).isoformat() if ts.iloc[i] > 1e12 else datetime.fromtimestamp(ts.iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(entry_price),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(entry_price),
                    'raw_price_b': float(entry_price)
                })
                trade_num += 1

    return entries