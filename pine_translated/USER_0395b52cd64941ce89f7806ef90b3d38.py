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
    # Constants from script
    i_htfSwingN = 20
    i_ltfSwingN = 10
    i_sweepLB = 5
    i_fvgProx = 0.5
    i_mode = "Day Trade"
    isSwing = i_mode == "Swing"
    fibEntry = 0.67 if not isSwing else 0.71
    i_reqSweep = True
    i_reqFVG = True
    i_useSession = False
    bufTick = 0.0  # simplified

    # Resample to 4H for HTF analysis
    df_4h = df.resample('4h', on='time', origin='start').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna(subset=['high']).reset_index()

    if len(df_4h) < i_htfSwingN * 2:
        return []

    # HTF pivot high/low using rolling window
    def calc_swing_highs(series, length):
        pivot = series.rolling(window=length * 2, min_periods=1).apply(lambda x: x.iloc[-1] if len(x) >= length else np.nan, raw=True)
        result = pd.Series(np.nan, index=series.index)
        for i in range(length - 1, len(series)):
            window = series.iloc[i - length + 1:i + 1]
            if window.idxmax() == series.index[i]:
                result.iloc[i] = series.iloc[i]
        return result

    def calc_swing_lows(series, length):
        result = pd.Series(np.nan, index=series.index)
        for i in range(length - 1, len(series)):
            window = series.iloc[i - length + 1:i + 1]
            if window.idxmin() == series.index[i]:
                result.iloc[i] = series.iloc[i]
        return result

    htfPH_series = calc_swing_highs(df_4h['high'], i_htfSwingN)
    htfPL_series = calc_swing_lows(df_4h['low'], i_htfSwingN)

    # HTF swing high/low latching
    htfSwingHigh_arr = pd.Series(np.nan, index=df_4h.index)
    htfSwingLow_arr = pd.Series(np.nan, index=df_4h.index)
    for i in range(len(df_4h)):
        if not pd.isna(htfPH_series.iloc[i]):
            htfSwingHigh_arr.iloc[i] = htfPH_series.iloc[i]
        else:
            htfSwingHigh_arr.iloc[i] = htfSwingHigh_arr.iloc[i-1] if i > 0 else np.nan
        if not pd.isna(htfPL_series.iloc[i]):
            htfSwingLow_arr.iloc[i] = htfPL_series.iloc[i]
        else:
            htfSwingLow_arr.iloc[i] = htfSwingLow_arr.iloc[i-1] if i > 0 else np.nan

    # HTF mid and bias
    htfMid_arr = pd.Series(np.nan, index=df_4h.index)
    bullBias_arr = pd.Series(False, index=df_4h.index)
    bearBias_arr = pd.Series(False, index=df_4h.index)
    for i in range(len(df_4h)):
        hs = htfSwingHigh_arr.iloc[i]
        ls = htfSwingLow_arr.iloc[i]
        if not pd.isna(hs) and not pd.isna(ls):
            mid = (hs + ls) / 2.0
            htfMid_arr.iloc[i] = mid
            htfClose = df_4h['close'].iloc[i]
            bullBias_arr.iloc[i] = htfClose < mid
            bearBias_arr.iloc[i] = htfClose > mid

    # LTF swing high/low
    ltfSwingHigh_arr = calc_swing_highs(df['high'], i_ltfSwingN)
    ltfSwingLow_arr = calc_swing_lows(df['low'], i_ltfSwingN)

    # Sweep reference (shifted by 1 for [1])
    sweepRefHigh = df['high'].shift(1).rolling(window=i_sweepLB).max()
    sweepRefLow = df['low'].shift(1).rolling(window=i_sweepLB).min()

    # Bear/Bull sweep signals
    bearSweep_sig = (df['high'] > sweepRefHigh) & (df['close'] < sweepRefHigh)
    bullSweep_sig = (df['low'] < sweepRefLow) & (df['close'] > sweepRefLow)

    # FVG detection (shifted by 2 for [2])
    bearFVG_sig = df['high'].shift(2) < df['low']
    bullFVG_sig = df['low'].shift(2) > df['high']

    # Initialize latched variables
    lastBearSweep_arr = pd.Series(False, index=df.index)
    lastBullSweep_arr = pd.Series(False, index=df.index)
    lastBearSweepPx_arr = pd.Series(np.nan, index=df.index)
    lastBullSweepPx_arr = pd.Series(np.nan, index=df.index)
    bearSweepBar_arr = pd.Series(np.nan, index=df.index)
    bullSweepBar_arr = pd.Series(np.nan, index=df.index)
    latBearFVGTop_arr = pd.Series(np.nan, index=df.index)
    latBearFVGBot_arr = pd.Series(np.nan, index=df.index)
    latBullFVGTop_arr = pd.Series(np.nan, index=df.index)
    latBullFVGBot_arr = pd.Series(np.nan, index=df.index)
    bearFVGBar_arr = pd.Series(np.nan, index=df.index)
    bullFVGBar_arr = pd.Series(np.nan, index=df.index)
    dispBearLow_arr = pd.Series(np.nan, index=df.index)
    dispBullHigh_arr = pd.Series(np.nan, index=df.index)

    # Track displacement extremes
    for i in range(len(df)):
        if i > 0:
            prevBearSweep = bearSweepBar_arr.iloc[i-1]
            prevBullSweep = bullSweepBar_arr.iloc[i-1]
            prevDispBearLow = dispBearLow_arr.iloc[i-1]
            prevDispBullHigh = dispBullHigh_arr.iloc[i-1]
            if not pd.isna(prevBearSweep) and i > prevBearSweep:
                low_val = df['low'].iloc[i]
                dispBearLow_arr.iloc[i] = low_val if pd.isna(prevDispBearLow) else min(prevDispBearLow, low_val)
            else:
                dispBearLow_arr.iloc[i] = prevDispBearLow
            if not pd.isna(prevBullSweep) and i > prevBullSweep:
                high_val = df['high'].iloc[i]
                dispBullHigh_arr.iloc[i] = high_val if pd.isna(prevDispBullHigh) else max(prevDispBullHigh, high_val)
            else:
                dispBullHigh_arr.iloc[i] = prevDispBullHigh
        if bearSweep_sig.iloc[i] and bearBias_arr.iloc[i]:
            lastBearSweep_arr.iloc[i] = True
            lastBearSweepPx_arr.iloc[i] = df['high'].iloc[i]
            bearSweepBar_arr.iloc[i] = float(i)
        if bullSweep_sig.iloc[i] and bullBias_arr.iloc[i]:
            lastBullSweep_arr.iloc[i] = True
            lastBullSweepPx_arr.iloc[i] = df['low'].iloc[i]
            bullSweepBar_arr.iloc[i] = float(i)
        if bearFVG_sig.iloc[i]:
            latBearFVGTop_arr.iloc[i] = df['low'].iloc[i]
            latBearFVGBot_arr.iloc[i] = df['high'].shift(2).iloc[i]
            bearFVGBar_arr.iloc[i] = float(i)
        if bullFVG_sig.iloc[i]:
            latBullFVGTop_arr.iloc[i] = df['low'].shift(2).iloc[i]
            latBullFVGBot_arr.iloc[i] = df['high'].iloc[i]
            bullFVGBar_arr.iloc[i] = float(i)

    # BoS detection
    bearBoS_sig = pd.Series(False, index=df.index)
    bullBoS_sig = pd.Series(False, index=df.index)
    for i in range(1, len(df)):
        ltfSL = ltfSwingLow_arr.iloc[i]
        ltfSH = ltfSwingHigh_arr.iloc[i]
        bsb = bearSweepBar_arr.iloc[i-1] if i > 0 else np.nan
        bullSB = bullSweepBar_arr.iloc[i-1] if i > 0 else np.nan
        if not pd.isna(ltfSL) and df['open'].iloc[i] < ltfSL and df['close'].iloc[i] < ltfSL and not pd.isna(bsb) and i > bsb:
            bearBoS_sig.iloc[i] = True
        if not pd.isna(ltfSH) and df['open'].iloc[i] > ltfSH and df['close'].iloc[i] > ltfSH and not pd.isna(bullSB) and i > bullSB:
            bullBoS_sig.iloc[i] = True

    # Fib level function
    def f_fibLevel(pivHigh, pivLow, pct, isBull):
        if isBull:
            return pivLow + pct * (pivHigh - pivLow)
        else:
            return pivHigh - pct * (pivHigh - pivLow)

    # FVG confluence check
    def f_fvgConfl(entryLvl, fvgTop, fvgBot, prox_pct=0.5):
        proxAmt = entryLvl * (prox_pct / 100.0)
        return entryLvl >= (fvgBot - proxAmt) and entryLvl <= (fvgTop + proxAmt)

    # Session filter (simplified - always true if not used)
    inSession = pd.Series(True, index=df.index)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if not inSession.iloc[i]:
            continue

        # Bear entry conditions
        if bearBias_arr.iloc[i] and bearBoS_sig.iloc[i]:
            cond_sweep = not i_reqSweep or lastBearSweep_arr.iloc[i]
            if not cond_sweep:
                continue

            # FVG check: must exist and between sweep bar and current bar
            fvgOk = not i_reqFVG or (not pd.isna(latBearFVGTop_arr.iloc[i]) and not pd.isna(bearFVGBar_arr.iloc[i]) and not pd.isna(bearSweepBar_arr.iloc[i]) and bearFVGBar_arr.iloc[i] >= bearSweepBar_arr.iloc[i] and bearFVGBar_arr.iloc[i] <= i)

            if not fvgOk:
                continue

            fibH_val = lastBearSweepPx_arr.iloc[i]
            fibL_val = dispBearLow_arr.iloc[i]

            if pd.isna(fibH_val) or pd.isna(fibL_val) or fibH_val <= fibL_val:
                continue

            entryLvl = f_fibLevel(fibH_val, fibL_val, fibEntry, False)

            # FVG confluence at entry level
            fvgConfl = not i_reqFVG or f_fvgConfl(entryLvl, latBearFVGTop_arr.iloc[i], latBearFVGBot_arr.iloc[i], i_fvgProx)

            if not fvgConfl:
                continue

            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000.0, tz=timezone.utc).isoformat(),
                'entry_price_guess': entryLvl,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entryLvl,
                'raw_price_b': entryLvl
            })
            trade_num += 1

        # Bull entry conditions
        if bullBias_arr.iloc[i] and bullBoS_sig.iloc[i]:
            cond_sweep = not i_reqSweep or lastBullSweep_arr.iloc[i]
            if not cond_sweep:
                continue

            fvgOk = not i_reqFVG or (not pd.isna(latBullFVGTop_arr.iloc[i]) and not pd.isna(bullFVGBar_arr.iloc[i]) and not pd.isna(bullSweepBar_arr.iloc[i]) and bullFVGBar_arr.iloc[i] >= bullSweepBar_arr.iloc[i] and bullFVGBar_arr.iloc[i] <= i)

            if not fvgOk:
                continue

            fibH_val = dispBullHigh_arr.iloc[i]
            fibL_val = lastBullSweepPx_arr.iloc[i]

            if pd.isna(fibH_val) or pd.isna(fibL_val) or fibH_val <= fibL_val:
                continue

            entryLvl = f_fibLevel(fibH_val, fibL_val, fibEntry, True)

            fvgConfl = not i_reqFVG or f_fvgConfl(entryLvl, latBullFVGTop_arr.iloc[i], latBullFVGBot_arr.iloc[i], i_fvgProx)

            if not fvgConfl:
                continue

            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000.0, tz=timezone.utc).isoformat(),
                'entry_price_guess': entryLvl,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entryLvl,
                'raw_price_b': entryLvl
            })
            trade_num += 1

    return entries