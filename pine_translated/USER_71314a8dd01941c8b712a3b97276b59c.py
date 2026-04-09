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
    # Input parameters (using defaults from script)
    i_left = 20
    i_right = 15
    i_atrLen = 14
    i_mult = 0.5
    i_emaFast = 20
    i_emaSlow = 50
    i_bandFilter = "EMA"
    i_bbLen = 20
    i_bbMult = 2.0
    i_hamRatio = 33.0
    i_hamShadow = 5.0
    i_minSize = 0.3
    i_useEngulf = True
    i_useHammer = True
    i_longs = True
    i_shorts = True
    i_useSession = True
    session_start = "07:00"
    session_end = "16:00"

    o = df['open'].copy()
    h = df['high'].copy()
    l = df['low'].copy()
    c = df['close'].copy()
    t = df['time'].copy()

    # ── Wilder ATR ──────────────────────────────────────────────
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/i_atrLen, adjust=False).mean()
    band = atr * i_mult / 2

    # ── Heikin Ashi ─────────────────────────────────────────────
    haClose = (o + h + l + c) / 4
    haOpen = pd.Series(index=o.index, dtype=float)
    haOpen.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2
    for i in range(1, len(haOpen)):
        haOpen.iloc[i] = (haOpen.iloc[i-1] + haClose.iloc[i-1]) / 2
    hiHaBod = pd.concat([haClose, haOpen], axis=1).max(axis=1)
    loHaBod = pd.concat([haClose, haOpen], axis=1).min(axis=1)

    # ── Pivot Highs/Lows ────────────────────────────────────────
    phSrc = hiHaBod
    plSrc = loHaBod
    ph = pd.Series(index=phSrc.index, dtype=float)
    pl = pd.Series(index=plSrc.index, dtype=float)

    for i in range(i_left + i_right, len(phSrc)):
        left_window = phSrc.iloc[i - i_left:i]
        right_window = phSrc.iloc[i + 1:i + i_right + 1]
        if len(left_window) == i_left and len(right_window) == i_right:
            if left_window.max() <= phSrc.iloc[i] and right_window.max() <= phSrc.iloc[i]:
                ph.iloc[i] = phSrc.iloc[i]

    for i in range(i_left + i_right, len(plSrc)):
        left_window = plSrc.iloc[i - i_left:i]
        right_window = plSrc.iloc[i + 1:i + i_right + 1]
        if len(left_window) == i_left and len(right_window) == i_right:
            if left_window.min() >= plSrc.iloc[i] and right_window.min() >= plSrc.iloc[i]:
                pl.iloc[i] = plSrc.iloc[i]

    # ── Support/Resistance Levels ───────────────────────────────
    pivRes = []
    pivSup = []
    resTop = pd.Series(index=h.index, dtype=float)
    resBtm = pd.Series(index=h.index, dtype=float)
    supTop = pd.Series(index=h.index, dtype=float)
    supBtm = pd.Series(index=h.index, dtype=float)

    resTop.iloc[:] = np.nan
    resBtm.iloc[:] = np.nan
    supTop.iloc[:] = np.nan
    supBtm.iloc[:] = np.nan

    for i in range(len(h)):
        if i >= i_right:
            band_val = band.iloc[i - i_right] if i - i_right >= 0 else band.iloc[0]
            if not pd.isna(ph.iloc[i]):
                new_res = ph.iloc[i] + band_val
                pivRes.insert(0, new_res)
                if len(pivRes) > 3:
                    pivRes.pop()
            if not pd.isna(pl.iloc[i]):
                new_sup = pl.iloc[i] - band_val
                pivSup.insert(0, new_sup)
                if len(pivSup) > 3:
                    pivSup.pop()

        nearestRes = pivRes[0] if len(pivRes) > 0 else np.nan
        nearestSup = pivSup[0] if len(pivSup) > 0 else np.nan

        resTop.iloc[i] = nearestRes
        resBtm.iloc[i] = nearestRes - band.iloc[i] if not pd.isna(nearestRes) else np.nan
        supTop.iloc[i] = nearestSup + band.iloc[i] if not pd.isna(nearestSup) else np.nan
        supBtm.iloc[i] = nearestSup

    # ── EMA Band ────────────────────────────────────────────────
    emaFast = c.ewm(span=i_emaFast, adjust=False).mean()
    emaSlow = c.ewm(span=i_emaSlow, adjust=False).mean()
    emaUpper = pd.concat([emaFast, emaSlow], axis=1).max(axis=1)
    emaLower = pd.concat([emaFast, emaSlow], axis=1).min(axis=1)
    inEmaBull = (c > emaSlow) & (emaFast > emaSlow)
    inEmaBear = (c < emaSlow) & (emaFast < emaSlow)
    inEmaBand = (c < emaUpper) & (c > emaLower)

    # ── Bollinger Bands ─────────────────────────────────────────
    bbMid = c.rolling(i_bbLen).mean()
    bbStd = c.rolling(i_bbLen).std()
    bbUpp = bbMid + i_bbMult * bbStd
    bbLow = bbMid - i_bbMult * bbStd
    inBbBull = (c <= bbMid) & (c >= bbLow)
    inBbBear = (c >= bbMid) & (c <= bbUpp)

    # ── Band filter ──────────────────────────────────────────────
    if i_bandFilter == "EMA":
        bandBull = inEmaBull
        bandBear = inEmaBear
    elif i_bandFilter == "Bollinger":
        bandBull = (c > emaSlow) & inBbBull
        bandBear = (c < emaSlow) & inBbBear
    else:
        bandBull = pd.Series(True, index=c.index)
        bandBear = pd.Series(True, index=c.index)

    # ── Session Filter ──────────────────────────────────────────
    if i_useSession:
        dt_series = pd.to_datetime(t, unit='s', utc=True)
        time_str = dt_series.dt.strftime('%H:%M')
        inSession = (time_str >= session_start) & (time_str <= session_end)
    else:
        inSession = pd.Series(True, index=c.index)

    # ── Candle helpers ──────────────────────────────────────────
    bodySize = (c - o).abs()
    candleRange = h - l
    upperWick = h - pd.concat([c, o], axis=1).max(axis=1)
    lowerWick = pd.concat([c, o], axis=1).min(axis=1) - l
    isBullCandle = c > o
    isBearCandle = c < o

    bodyRatio = pd.Series(np.where(candleRange > 0, bodySize / candleRange * 100, 0), index=c.index)
    lowerRatio = pd.Series(np.where(candleRange > 0, lowerWick / candleRange * 100, 0), index=c.index)
    upperRatio = pd.Series(np.where(candleRange > 0, upperWick / candleRange * 100, 0), index=c.index)
    bigEnough = candleRange >= i_minSize * atr

    # ── Hammer (bullish) ─────────────────────────────────────────
    isHammer = (bigEnough &
                (bodyRatio <= i_hamRatio) &
                (lowerRatio >= (100 - i_hamRatio) * 0.6) &
                (upperWick <= bodySize * (i_hamShadow / 100 + 1)))

    # ── Shooting Star (bearish) ─────────────────────────────────
    isStar = (bigEnough &
              (bodyRatio <= i_hamRatio) &
              (upperRatio >= (100 - i_hamRatio) * 0.6) &
              (lowerWick <= bodySize * (i_hamShadow / 100 + 1)))

    # ── Bullish Engulfing ────────────────────────────────────────
    isBullEngulf = (isBullCandle &
                    (c > h.shift(1)) &
                    (o < l.shift(1)) &
                    isBearCandle.shift(1) &
                    bigEnough)

    # ── Bearish Engulfing ───────────────────────────────────────
    isBearEngulf = (isBearCandle &
                    (c < l.shift(1)) &
                    (o > h.shift(1)) &
                    isBullCandle.shift(1) &
                    bigEnough)

    # ── At-level detection ──────────────────────────────────────
    atSupport = (pd.notna(supTop) & pd.notna(supBtm) &
                 (l <= supTop) & (h >= supBtm))
    atResist = (pd.notna(resTop) & pd.notna(resBtm) &
                (h >= resBtm) & (l <= resTop))

    # ── Rejection signals ───────────────────────────────────────
    bullRej = (atSupport &
               ((i_useHammer & isHammer) | (i_useEngulf & isBullEngulf)))
    bearRej = (atResist &
               ((i_useHammer & isStar) | (i_useEngulf & isBearEngulf)))

    # ── Confirmed signals (skip first 2 bars for shift/prev refs) ──
    bullSig = bullRej & bandBull & inSession
    bearSig = bearRej & bandBear & inSession
    bullSig.iloc[:i_right + 1] = False
    bearSig.iloc[:i_right + 1] = False

    # ── Generate entries ─────────────────────────────────────────
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i < i_right + 1:
            continue
        if pd.isna(atr.iloc[i]):
            continue

        if bullSig.iloc[i] and i_longs:
            entry_price = c.iloc[i]
            ts = int(t.iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        if bearSig.iloc[i] and i_shorts:
            entry_price = c.iloc[i]
            ts = int(t.iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
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