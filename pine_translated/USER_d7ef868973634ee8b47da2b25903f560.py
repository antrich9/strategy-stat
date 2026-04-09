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

    atrLength = 14
    atrMultiplier = 3.0
    takeProfitRatio = 1.5
    lengthMD = 10
    tradeDirection = "Both"

    chandelierExitLength = 22
    chandelierMultiplier = 3.0

    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    input_repType = 'On'

    n = len(df)
    if n < input_lookback * 2 + 10:
        return []

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    time_col = df['time'].values

    # Wilder ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = high[0] - low[0]
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * (atrLength - 1) + tr[i]) / atrLength

    # Chandelier Exit
    highestHigh = pd.Series(high).rolling(chandelierExitLength).max().values
    lowestLow = pd.Series(low).rolling(chandelierExitLength).min().values
    chandelierExitLong = highestHigh - (atr * chandelierMultiplier)
    chandelierExitShort = lowestLow + (atr * chandelierMultiplier)

    # Pivot points
    bb = input_lookback
    pl = np.full(n, np.nan)
    ph = np.full(n, np.nan)

    for i in range(bb, n - bb):
        is_pl = True
        for j in range(i - bb, i + bb + 1):
            if j != i and low[j] <= low[i]:
                is_pl = False
                break
        if is_pl:
            pl[i] = low[i]

        is_ph = True
        for j in range(i - bb, i + bb + 1):
            if j != i and high[j] >= high[i]:
                is_ph = False
                break
        if is_ph:
            ph[i] = high[i]

    pl_changed = np.zeros(n, dtype=bool)
    ph_changed = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(pl[i]) and (np.isnan(pl[i-1]) or pl[i] != pl[i-1]):
            pl_changed[i] = True
        if not np.isnan(ph[i]) and (np.isnan(ph[i-1]) or ph[i] != ph[i-1]):
            ph_changed[i] = True

    # Box boundaries
    sTop = np.full(n, np.nan)
    sBot = np.full(n, np.nan)
    rTop = np.full(n, np.nan)
    rBot = np.full(n, np.nan)

    sBreak = np.zeros(n, dtype=bool)
    rBreak = np.zeros(n, dtype=bool)

    sRetValid = np.zeros(n, dtype=bool)
    rRetValid = np.zeros(n, dtype=bool)

    sRetEvent = np.zeros(n, dtype=bool)
    rRetEvent = np.zeros(n, dtype=bool)

    sBreak_active = False
    rBreak_active = False
    sTop_val = np.nan
    sBot_val = np.nan
    rTop_val = np.nan
    rBot_val = np.nan
    sBreak_idx = 0
    rBreak_idx = 0

    sRetOccurred = False
    rRetOccurred = False
    sRetEvent_bar = 0
    rRetEvent_bar = 0
    sRetValue = np.nan
    rRetValue = np.nan
    sRetSince = 0
    rRetSince = 0

    for i in range(1, n):
        if pl_changed[i]:
            if not sBreak_active:
                if not np.isnan(sBot[i-1]):
                    sBot[i-1] = np.nan
            sBreak_active = False
            sRetOccurred = False

        if ph_changed[i]:
            if not rBreak_active:
                if not np.isnan(rTop[i-1]):
                    rTop[i-1] = np.nan
            rBreak_active = False
            rRetOccurred = False

        cu = close[i] < sBot[i] and close[i-1] >= sBot[i] if not np.isnan(sBot[i]) else False
        co = close[i] > rTop[i] and close[i-1] <= rTop[i] if not np.isnan(rTop[i]) else False

        if cu and not sBreak_active:
            sBreak_active = True
            sBreak_idx = i
            sBreak[i] = True
            sTop_val = sTop[i]
            sBot_val = sBot[i]

        if co and not rBreak_active:
            rBreak_active = True
            rBreak_idx = i
            rBreak[i] = True
            rTop_val = rTop[i]
            rBot_val = rBot[i]

        if pl_changed[i]:
            s_yLoc_val = low[i-1] if low[i+1] > low[i-1] else low[i+1]
            sTop[i] = s_yLoc_val
            sBot[i] = pl[i]
        elif sBreak_active:
            sTop[i] = sTop_val
            sBot[i] = sBot_val

        if ph_changed[i]:
            r_yLoc_val = high[i-1] if high[i+1] > high[i-1] else high[i+1]
            rTop[i] = ph[i]
            rBot[i] = r_yLoc_val
        elif rBreak_active:
            rTop[i] = rTop_val
            rBot[i] = rBot_val

        # Retest conditions for support
        bars_since_sBreak = i - sBreak_idx if sBreak_active else 999
        sRetActive = False
        if sBreak_active and bars_since_sBreak > input_retSince and bars_since_sBreak <= input_retSince + input_retValid:
            st = sTop_val
            sb = sBot_val
            if (high[i] >= st and close[i] <= sb) or \
               (high[i] >= st and close[i] >= sb and close[i] <= st) or \
               (high[i] >= sb and high[i] <= st) or \
               (high[i] >= sb and high[i] <= st and close[i] < sb):
                sRetActive = True

        if sRetActive and not sRetEvent[i-1]:
            sRetEvent[i] = True
            sRetOccurred = False
            sRetEvent_bar = i
            sRetValue = st
            sRetSince = 0
        else:
            sRetEvent[i] = sRetEvent[i-1]

        if sRetEvent[i]:
            sRetSince = i - sRetEvent_bar

        if input_repType == 'On':
            sRetConditions = close[i] <= sRetValue
        elif input_repType == 'Off: High & Low':
            sRetConditions = low[i] <= sRetValue
        else:
            sRetConditions = close[i] <= sRetValue

        if sRetEvent[i] and sRetSince > 0 and sRetSince <= input_retValid and sRetConditions and not sRetOccurred:
            sRetValid[i] = True
            sRetOccurred = True

        if sRetEvent[i] and sRetSince > input_retValid:
            sRetEvent[i] = False
            sRetOccurred = False

        # Retest conditions for resistance
        bars_since_rBreak = i - rBreak_idx if rBreak_active else 999
        rRetActive = False
        if rBreak_active and bars_since_rBreak > input_retSince and bars_since_rBreak <= input_retSince + input_retValid:
            rt = rTop_val
            rb = rBot_val
            if (low[i] <= rb and close[i] >= rt) or \
               (low[i] <= rb and close[i] <= rt and close[i] >= rb) or \
               (low[i] <= rt and low[i] >= rb) or \
               (low[i] <= rt and low[i] >= rb and close[i] > rt):
                rRetActive = True

        if rRetActive and not rRetEvent[i-1]:
            rRetEvent[i] = True
            rRetOccurred = False
            rRetEvent_bar = i
            rRetValue = rt
            rRetSince = 0
        else:
            rRetEvent[i] = rRetEvent[i-1]

        if rRetEvent[i]:
            rRetSince = i - rRetEvent_bar

        if input_repType == 'On':
            rRetConditions = close[i] >= rRetValue
        elif input_repType == 'Off: High & Low':
            rRetConditions = high[i] >= rRetValue
        else:
            rRetConditions = close[i] >= rRetValue

        if rRetEvent[i] and rRetSince > 0 and rRetSince <= input_retValid and rRetConditions and not rRetOccurred:
            rRetValid[i] = True
            rRetOccurred = True

        if rRetEvent[i] and rRetSince > input_retValid:
            rRetEvent[i] = False
            rRetOccurred = False

        # Clean up boxes when pivots change
        if pl_changed[i] and sRetOccurred:
            sRetOccurred = False
        if ph_changed[i] and rRetOccurred:
            rRetOccurred = False

    entries = []
    trade_num = 1

    for i in range(n):
        if np.isnan(close[i]):
            continue

        if sRetValid[i] and (tradeDirection == "Long" or tradeDirection == "Both"):
            ts = int(time_col[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close[i]),
                'raw_price_b': float(close[i])
            })
            trade_num += 1

        if rRetValid[i] and (tradeDirection == "Short" or tradeDirection == "Both"):
            ts = int(time_col[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close[i]),
                'raw_price_b': float(close[i])
            })
            trade_num += 1

    return entries