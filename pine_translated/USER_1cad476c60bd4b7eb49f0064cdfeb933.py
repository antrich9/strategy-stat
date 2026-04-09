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
    entries = []
    trade_num = 1

    # Default input values from Pine Script
    useHTFBias = True
    htfTF = "240"
    htfPivotLen = 2
    pivotLen = 2
    bosLookback = 20
    useSweep = True
    sweepMaxBars = 30
    useFVG = True
    requireFibInFVG = False
    fibLevel = 0.71
    minScore = 50

    n = len(df)
    if n < 3:
        return entries

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    time_col = df['time'].values

    # --- Calculate HTF data ---
    htf_high = np.zeros(n)
    htf_low = np.zeros(n)
    htf_close = np.zeros(n)

    # Simple HTF resampling - use last HTF bar value for each bar
    # In real implementation, this would use request.security
    # Here we approximate by using the same timeframe data
    # For proper HTF, we'd need to upsample
    htf_high = high.copy()
    htf_low = low.copy()
    htf_close = close.copy()

    # --- HTF Pivot High/Low ---
    htfRangeHigh = np.nan
    htfRangeLow = np.nan
    htf_mid = np.zeros(n)
    bias = np.zeros(n)

    for i in range(htfPivotLen, n - htfPivotLen):
        # Pivothigh: highest high in window of length 2*pivotLen+1 centered at pivotLen
        window_start = i - htfPivotLen
        window_end = i + htfPivotLen + 1
        if window_end <= n:
            window = htf_high[window_start:window_end]
            ph_val = np.max(window)
            if ph_val == htf_high[i] and not np.isnan(ph_val):
                htfRangeHigh = ph_val

        # Pivotlow: lowest low in window
        window = htf_low[window_start:window_end]
        pl_val = np.min(window)
        if pl_val == htf_low[i] and not np.isnan(pl_val):
            htfRangeLow = pl_val

        # HTF Mid
        if not np.isnan(htfRangeHigh) and not np.isnan(htfRangeLow):
            htf_mid[i] = (htfRangeHigh + htfRangeLow) / 2
        else:
            htf_mid[i] = htf_mid[i-1] if i > 0 else np.nan

        # Bias
        if useHTFBias:
            if not np.isnan(htf_close[i]) and not np.isnan(htf_mid[i]):
                if htf_close[i] < htf_mid[i]:
                    bias[i] = 1
                elif htf_close[i] > htf_mid[i]:
                    bias[i] = -1
                else:
                    bias[i] = 0

    # --- Local Structure ---
    lastPH = np.full(n, np.nan)
    lastPL = np.full(n, np.nan)

    for i in range(pivotLen, n - pivotLen):
        window_start = i - pivotLen
        window_end = i + pivotLen + 1
        if window_end <= n:
            window = high[window_start:window_end]
            ph_val = np.max(window)
            if ph_val == high[i] and not np.isnan(ph_val):
                lastPH[i] = ph_val

            window = low[window_start:window_end]
            pl_val = np.min(window)
            if pl_val == low[i] and not np.isnan(pl_val):
                lastPL[i] = pl_val

    # Forward fill lastPH and lastPL
    lastPH_filled = np.full(n, np.nan)
    lastPL_filled = np.full(n, np.nan)
    current_ph = np.nan
    current_pl = np.nan

    for i in range(n):
        if not np.isnan(lastPH[i]):
            current_ph = lastPH[i]
        if not np.isnan(lastPL[i]):
            current_pl = lastPL[i]
        lastPH_filled[i] = current_ph
        lastPL_filled[i] = current_pl

    # --- BOS Detection ---
    bosBull = np.zeros(n, dtype=bool)
    bosBear = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if close[i] > lastPH_filled[i] and close[i-1] <= lastPH_filled[i-1]:
            bosBull[i] = True
        if close[i] < lastPL_filled[i] and close[i-1] >= lastPL_filled[i-1]:
            bosBear[i] = True

    # --- BOS tracking variables ---
    lastBosBullBar = np.zeros(n, dtype=int)
    lastBosBearBar = np.zeros(n, dtype=int)
    bosImpulseHigh = np.full(n, np.nan)
    bosImpulseLow = np.full(n, np.nan)

    for i in range(n):
        if bosBull[i]:
            lastBosBullBar[i] = i
            if i > 0:
                lastBosBullBar[i] = i
        elif i > 0:
            lastBosBullBar[i] = lastBosBullBar[i-1]

        if bosBear[i]:
            lastBosBearBar[i] = i
        elif i > 0:
            lastBosBearBar[i] = lastBosBearBar[i-1]

    for i in range(n):
        if bosBull[i]:
            bosImpulseLow[i] = lastPL_filled[i] if not np.isnan(lastPL_filled[i]) else np.nan
            bosImpulseHigh[i] = close[i]
        elif i > 0:
            if not np.isnan(bosImpulseLow[i-1]):
                bosImpulseLow[i] = bosImpulseLow[i-1]
            if not np.isnan(bosImpulseHigh[i-1]):
                bosImpulseHigh[i] = bosImpulseHigh[i-1]

        if bosBear[i]:
            bosImpulseHigh[i] = lastPH_filled[i] if not np.isnan(lastPH_filled[i]) else np.nan
            bosImpulseLow[i] = close[i]
        elif i > 0:
            if not np.isnan(bosImpulseHigh[i-1]):
                bosImpulseHigh[i] = bosImpulseHigh[i-1]
            if not np.isnan(bosImpulseLow[i-1]):
                bosImpulseLow[i] = bosImpulseLow[i-1]

    # --- Liquidity Sweep ---
    sweptLow = np.zeros(n, dtype=bool)
    sweptHigh = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if not np.isnan(lastPL_filled[i]):
            for j in range(1, min(sweepMaxBars + 1, i + 1)):
                if i - j >= 0:
                    if low[i-j] < lastPL_filled[i-j] and close[i-j] > lastPL_filled[i-j]:
                        sweptLow[i] = True
                        break

        if not np.isnan(lastPH_filled[i]):
            for j in range(1, min(sweepMaxBars + 1, i + 1)):
                if i - j >= 0:
                    if high[i-j] > lastPH_filled[i-j] and close[i-j] < lastPH_filled[i-j]:
                        sweptHigh[i] = True
                        break

    # --- FVG Detection ---
    isBullFVG = np.zeros(n, dtype=bool)
    isBearFVG = np.zeros(n, dtype=bool)
    lastBullFVGTop = np.full(n, np.nan)
    lastBullFVGBot = np.full(n, np.nan)
    lastBearFVGTop = np.full(n, np.nan)
    lastBearFVGBot = np.full(n, np.nan)

    for i in range(2, n):
        if low[i] > high[i-2]:
            isBullFVG[i] = True
            lastBullFVGTop[i] = low[i-1]
            lastBullFVGBot[i] = high[i-2]
        if high[i] < low[i-2]:
            isBearFVG[i] = True
            lastBearFVGTop[i] = low[i-2]
            lastBearFVGBot[i] = high[i-1]

    # Forward fill FVG values
    for i in range(1, n):
        if np.isnan(lastBullFVGTop[i]):
            lastBullFVGTop[i] = lastBullFVGTop[i-1]
        if np.isnan(lastBullFVGBot[i]):
            lastBullFVGBot[i] = lastBullFVGBot[i-1]
        if np.isnan(lastBearFVGTop[i]):
            lastBearFVGTop[i] = lastBearFVGTop[i-1]
        if np.isnan(lastBearFVGBot[i]):
            lastBearFVGBot[i] = lastBearFVGBot[i-1]

    # --- Entry Conditions ---
    longFibEntry = np.full(n, np.nan)
    shortFibEntry = np.full(n, np.nan)

    for i in range(n):
        if not np.isnan(bosImpulseLow[i]) and not np.isnan(bosImpulseHigh[i]):
            longFibEntry[i] = bosImpulseLow[i] + (bosImpulseHigh[i] - bosImpulseLow[i]) * (1 - fibLevel)
        if not np.isnan(bosImpulseHigh[i]) and not np.isnan(bosImpulseLow[i]):
            shortFibEntry[i] = bosImpulseHigh[i] - (bosImpulseHigh[i] - bosImpulseLow[i]) * (1 - fibLevel)

    fibInBullFVG = np.zeros(n, dtype=bool)
    fibInBearFVG = np.zeros(n, dtype=bool)

    for i in range(n):
        if not np.isnan(longFibEntry[i]) and not np.isnan(lastBullFVGTop[i]) and not np.isnan(lastBullFVGBot[i]):
            if longFibEntry[i] <= lastBullFVGTop[i] and longFibEntry[i] >= lastBullFVGBot[i]:
                fibInBullFVG[i] = True
        if not np.isnan(shortFibEntry[i]) and not np.isnan(lastBearFVGTop[i]) and not np.isnan(lastBearFVGBot[i]):
            if shortFibEntry[i] <= lastBearFVGTop[i] and shortFibEntry[i] >= lastBearFVGBot[i]:
                fibInBearFVG[i] = True

    longScore = np.zeros(n)
    shortScore = np.zeros(n)

    for i in range(n):
        longScore[i] = (bias[i] >= 0 if useHTFBias else 25) + (25 if sweptLow[i] else 0) + (25 if isBullFVG[i] else 0) + (25 if fibInBullFVG[i] else 0)
        shortScore[i] = (bias[i] <= 0 if useHTFBias else 25) + (25 if sweptHigh[i] else 0) + (25 if isBearFVG[i] else 0) + (25 if fibInBearFVG[i] else 0)

    longCondition = np.zeros(n, dtype=bool)
    shortCondition = np.zeros(n, dtype=bool)

    for i in range(n):
        bars_since_bull = i - lastBosBullBar[i] if lastBosBullBar[i] > 0 else 999
        bars_since_bear = i - lastBosBearBar[i] if lastBosBearBar[i] > 0 else 999

        cond1 = bars_since_bull <= bosLookback
        cond2 = not useSweep or sweptLow[i]
        cond3 = not useFVG or isBullFVG[i]
        cond4 = not requireFibInFVG or fibInBullFVG[i]
        cond5 = not useHTFBias or bias[i] >= 0
        cond6 = longScore[i] >= minScore

        if cond1 and cond2 and cond3 and cond4 and cond5 and cond6:
            longCondition[i] = True

        cond1_s = bars_since_bear <= bosLookback
        cond2_s = not useSweep or sweptHigh[i]
        cond3_s = not useFVG or isBearFVG[i]
        cond4_s = not requireFibInFVG or fibInBearFVG[i]
        cond5_s = not useHTFBias or bias[i] <= 0
        cond6_s = shortScore[i] >= minScore

        if cond1_s and cond2_s and cond3_s and cond4_s and cond5_s and cond6_s:
            shortCondition[i] = True

    # --- Generate Entries ---
    in_position = False

    for i in range(n):
        if longCondition[i] and not in_position:
            entry_price = close[i]
            entry_ts = int(time_col[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat() if entry_ts > 10000000000 else datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()

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
            in_position = True

        elif shortCondition[i] and not in_position:
            entry_price = close[i]
            entry_ts = int(time_col[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat() if entry_ts > 10000000000 else datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()

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
            in_position = True

        # Reset position flag when not in a trade
        if not longCondition[i] and not shortCondition[i]:
            in_position = False

    return entries