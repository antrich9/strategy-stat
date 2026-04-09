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
    # Make a copy to avoid modifying original
    df = df.copy()

    # ─── GLITCH INDEX ─────────────────────────────────────────────────────────
    MaPeriod_glitch = 30
    price_glitch = (df['high'] + df['low']) / 2  # hl2
    sma_glitch = price_glitch.rolling(MaPeriod_glitch).mean()
    longEntry_Glitch = price_glitch > sma_glitch
    shortEntry_Glitch = price_glitch < sma_glitch
    GlitchSignalsLongFinal = longEntry_Glitch
    GlitchSignalsShortFinal = shortEntry_Glitch

    # ─── G-CHANNEL ─────────────────────────────────────────────────────────────
    src_gc = (df['high'] + df['low'] + df['close']) / 3  # hlc3
    N_gc = 4
    per_gc = 144
    mult_gc = 1.414
    modeLag_gc = False
    modeFast_gc = False
    crossGC = True
    inverseGC = False
    highlightMovementsGC = True

    # Calculate beta and alpha for G-Channel
    beta_gc = (1 - np.cos(4 * np.arcsin(1) / per_gc)) / (np.power(1.414, 2 / N_gc) - 1)
    alpha_gc = -beta_gc + np.sqrt(np.power(beta_gc, 2) + 2 * beta_gc)
    lag_gc = (per_gc - 1) / (2 * N_gc)

    # f_filt9x_gc implementation
    def f_filt9x_gc(a, s, i):
        f = np.zeros_like(s, dtype=float)
        m2_vals = {9: 36, 8: 28, 7: 21, 6: 15, 5: 10, 4: 6, 3: 3, 2: 1, 1: 0}
        m3_vals = {9: 84, 8: 56, 7: 35, 6: 20, 5: 10, 4: 4, 3: 1, 2: 0, 1: 0}
        m4_vals = {9: 126, 8: 70, 7: 35, 6: 15, 5: 5, 4: 1, 3: 0, 2: 0, 1: 0}
        m5_vals = {9: 126, 8: 56, 7: 21, 6: 6, 5: 1, 4: 0, 3: 0, 2: 0, 1: 0}
        m6_vals = {9: 84, 8: 28, 7: 7, 6: 1, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
        m7_vals = {9: 36, 8: 8, 7: 1, 6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
        m8_vals = {9: 9, 8: 1, 7: 0, 6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
        m9_vals = {9: 1, 8: 0, 7: 0, 6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
        m2 = m2_vals.get(i, 0)
        m3 = m3_vals.get(i, 0)
        m4 = m4_vals.get(i, 0)
        m5 = m5_vals.get(i, 0)
        m6 = m6_vals.get(i, 0)
        m7 = m7_vals.get(i, 0)
        m8 = m8_vals.get(i, 0)
        m9 = m9_vals.get(i, 0)
        x = 1 - a
        result = np.zeros_like(s, dtype=float)
        f_prev = np.zeros_like(s, dtype=float)
        f_prev2 = np.zeros_like(s, dtype=float)
        f_prev3 = np.zeros_like(s, dtype=float)
        f_prev4 = np.zeros_like(s, dtype=float)
        f_prev5 = np.zeros_like(s, dtype=float)
        f_prev6 = np.zeros_like(s, dtype=float)
        f_prev7 = np.zeros_like(s, dtype=float)
        f_prev8 = np.zeros_like(s, dtype=float)
        f_prev9 = np.zeros_like(s, dtype=float)
        for t in range(len(s)):
            if t >= 1:
                f_prev = result[t-1]
            if t >= 2:
                f_prev2 = result[t-2]
            if t >= 3:
                f_prev3 = result[t-3]
            if t >= 4:
                f_prev4 = result[t-4]
            if t >= 5:
                f_prev5 = result[t-5]
            if t >= 6:
                f_prev6 = result[t-6]
            if t >= 7:
                f_prev7 = result[t-7]
            if t >= 8:
                f_prev8 = result[t-8]
            if t >= 9:
                f_prev9 = result[t-9]
            s_val = s.iloc[t] if hasattr(s, 'iloc') else s[t]
            term1 = np.power(a, i) * s_val
            term2 = i * x * f_prev
            term3 = 0
            term4 = 0
            term5 = 0
            term6 = 0
            term7 = 0
            term8 = 0
            term9 = 0
            if i >= 2:
                term3 = m2 * np.power(x, 2) * f_prev2
            if i >= 3:
                term4 = m3 * np.power(x, 3) * f_prev3
            if i >= 4:
                term5 = m4 * np.power(x, 4) * f_prev4
            if i >= 5:
                term6 = m5 * np.power(x, 5) * f_prev5
            if i >= 6:
                term7 = m6 * np.power(x, 6) * f_prev6
            if i >= 7:
                term8 = m7 * np.power(x, 7) * f_prev7
            if i >= 8:
                term9 = m8 * np.power(x, 8) * f_prev8
            if i == 9:
                term9 = m9 * np.power(x, 9) * f_prev9
            result[t] = term1 + term2 - term3 + term4 - term5 + term6 - term7 + term8 - term9
        return pd.Series(result, index=s.index if hasattr(s, 'index') else None)

    # f_pole_gc implementation
    def f_pole_gc(a, s, i):
        f1 = f_filt9x_gc(a, s, 1)
        f2 = f_filt9x_gc(a, s, 2) if i >= 2 else pd.Series(np.zeros(len(s)), index=s.index)
        f3 = f_filt9x_gc(a, s, 3) if i >= 3 else pd.Series(np.zeros(len(s)), index=s.index)
        f4 = f_filt9x_gc(a, s, 4) if i >= 4 else pd.Series(np.zeros(len(s)), index=s.index)
        f5 = f_filt9x_gc(a, s, 5) if i >= 5 else pd.Series(np.zeros(len(s)), index=s.index)
        f6 = f_filt9x_gc(a, s, 6) if i >= 6 else pd.Series(np.zeros(len(s)), index=s.index)
        f7 = f_filt9x_gc(a, s, 7) if i >= 2 else pd.Series(np.zeros(len(s)), index=s.index)
        f8 = f_filt9x_gc(a, s, 8) if i >= 8 else pd.Series(np.zeros(len(s)), index=s.index)
        f9 = f_filt9x_gc(a, s, 9) if i == 9 else pd.Series(np.zeros(len(s)), index=s.index)
        fn = pd.Series(np.zeros(len(s)), index=s.index)
        for t in range(len(s)):
            if i == 1:
                fn.iloc[t] = f1.iloc[t]
            elif i == 2:
                fn.iloc[t] = f2.iloc[t]
            elif i == 3:
                fn.iloc[t] = f3.iloc[t]
            elif i == 4:
                fn.iloc[t] = f4.iloc[t]
            elif i == 5:
                fn.iloc[t] = f5.iloc[t]
            elif i == 6:
                fn.iloc[t] = f6.iloc[t]
            elif i == 7:
                fn.iloc[t] = f7.iloc[t]
            elif i == 8:
                fn.iloc[t] = f8.iloc[t]
            elif i == 9:
                fn.iloc[t] = f9.iloc[t]
        return fn, f1

    srcdata_gc = src_gc if not modeLag_gc else src_gc + src_gc - src_gc.shift(int(lag_gc))
    trdata_gc = (df['high'] - df['low']) if not modeLag_gc else 2 * (df['high'] - df['low']) - (df['high'] - df['low']).shift(int(lag_gc))

    filtn_gc, filt1_gc = f_pole_gc(alpha_gc, srcdata_gc, N_gc)
    filtntr_gc, filt1tr_gc = f_pole_gc(alpha_gc, trdata_gc, N_gc)

    filt_gc = (filtn_gc + filt1_gc) / 2 if modeFast_gc else filtn_gc
    filttr_gc = (filtntr_gc + filt1tr_gc) / 2 if modeFast_gc else filtntr_gc

    hband_gc = filt_gc + filttr_gc * mult_gc
    lband_gc = filt_gc - filttr_gc * mult_gc

    GCSignals = pd.Series(np.where(filt_gc > filt_gc.shift(1), 1, -1), index=df.index)
    basicLongCondition_GC = (GCSignals > 0) & (df['close'] > filt_gc)
    basicShortCondition_GC = (GCSignals < 0) & (df['close'] < filt_gc)

    GCSignalsLong = basicLongCondition_GC if highlightMovementsGC else (df['close'] > filt_gc)
    GCSignalsShort = basicShortCondition_GC if highlightMovementsGC else (df['close'] < filt_gc)

    GCSignalsLongPrev = GCSignalsLong.shift(1).fillna(False).astype(bool)
    GCSignalsShortPrev = GCSignalsShort.shift(1).fillna(False).astype(bool)
    GCSignalsLong = GCSignalsLong.astype(bool)
    GCSignalsShort = GCSignalsShort.astype(bool)

    GCSignalsLongCross = (~GCSignalsLongPrev) & GCSignalsLong if crossGC else GCSignalsLong
    GCSignalsShortCross = (~GCSignalsShortPrev) & GCSignalsShort if crossGC else GCSignalsShort

    GCSignalsLongFinal = GCSignalsShortCross if inverseGC else GCSignalsLongCross
    GCSignalsShortFinal = GCSignalsLongCross if inverseGC else GCSignalsShortCross

    # ─── VFI (Volume Flow Indicator) ───────────────────────────────────────────
    vfi_length = 130
    vfi_coef = 0.2
    vfi_volCutoff = 2.5
    vfi_smoothLen = 3
    vfi_smoothType = 'EMA'
    vfi_maLength = 30
    vfi_maType = 'SMA'
    vfi_almaOffset = 0.85
    vfi_almaSigma = 6.0

    avg = (df['high'] + df['low'] + df['close']) / 3
    inter = np.log(avg) - np.log(avg.shift(1))
    vInter = inter.rolling(30).std()
    cutOff = vfi_coef * vInter * df['close']
    vAve = df['volume'].shift(1).rolling(vfi_length).mean()
    vMax = vAve * vfi_volCutoff
    vC = df['volume'].clip(upper=vMax)
    mf = avg - avg.shift(1)
    vCp = pd.Series(np.where(mf > cutOff, vC, np.where(mf < -cutOff, -vC, 0)), index=df.index)
    sVfi = vCp.rolling(vfi_length).sum() / vAve

    if vfi_smoothType == 'EMA':
        value_vfi = sVfi.ewm(span=vfi_smoothLen, adjust=False).mean()
    elif vfi_smoothType == 'SMA':
        value_vfi = sVfi.rolling(vfi_smoothLen).mean()
    elif vfi_smoothType == 'WMA':
        value_vfi = sVfi.rolling(vfi_smoothLen).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
    elif vfi_smoothType == 'ALMA':
        m = int(round(vfi_almaOffset * (vfi_smoothLen - 1)))
        s = vfi_almaSigma
        w = np.exp(-np.arange(vfi_smoothLen)**2 / (2 * s**2))
        w = w / w.sum()
        value_vfi = sVfi.rolling(vfi_smoothLen).apply(lambda x: np.dot(x, w), raw=True)

    if vfi_maType == 'EMA':
        value_ma = value_vfi.ewm(span=vfi_maLength, adjust=False).mean()
    elif vfi_maType == 'SMA':
        value_ma = value_vfi.rolling(vfi_maLength).mean()
    elif vfi_maType == 'WMA':
        value_ma = value_vfi.rolling(vfi_maLength).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
    elif vfi_maType == 'ALMA':
        m = int(round(vfi_almaOffset * (vfi_maLength - 1)))
        s = vfi_almaSigma
        w = np.exp(-np.arange(vfi_maLength)**2 / (2 * s**2))
        w = w / w.sum()
        value_ma = value_vfi.rolling(vfi_maLength).apply(lambda x: np.dot(x, w), raw=True)

    vfiBullishCondition = (value_vfi > value_ma) & (value_vfi > 0)
    vfiBearishCondition = (value_vfi < value_ma) & (value_vfi < 0)

    # ─── COMBINED ENTRY CONDITIONS ──────────────────────────────────────────────
    bullishEntryCondition = GlitchSignalsLongFinal & GCSignalsLongFinal & vfiBullishCondition
    bearishEntryCondition = GlitchSignalsShortFinal & GCSignalsShortFinal & vfiBearishCondition

    # ─── GENERATE ENTRIES ───────────────────────────────────────────────────────
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i < max(MaPeriod_glitch, vfi_length + vfi_smoothLen + vfi_maLength, per_gc, 30):
            continue

        if bullishEntryCondition.iloc[i]:
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

        if bearishEntryCondition.iloc[i]:
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

    return entries