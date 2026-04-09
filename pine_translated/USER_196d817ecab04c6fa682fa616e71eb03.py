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

    close = df['close']
    high = df['high']
    low = df['low']
    open_col = df['open']
    volume = df['volume']
    hl2 = (high + low) / 2
    hlc3 = (high + low + close) / 3

    # ============ GLITCH LOGIC ============
    MaPeriod_glitch = 30
    price_glitch = hl2
    sma_glitch = price_glitch.rolling(30).mean()
    longEntry_Glitch = price_glitch > sma_glitch
    shortEntry_Glitch = price_glitch < sma_glitch
    GlitchSignalsLongFinal = longEntry_Glitch
    GlitchSignalsShortFinal = shortEntry_Glitch

    # ============ G-CHANNEL LOGIC ============
    N_gc = 4
    per_gc = 144
    mult_gc = 1.414
    modeLag_gc = False
    modeFast_gc = False
    src_gc = hlc3

    beta_gc = (1 - np.cos(4 * np.arcsin(1) / per_gc)) / (np.power(1.414, 2 / N_gc) - 1)
    alpha_gc = -beta_gc + np.sqrt(np.power(beta_gc, 2) + 2 * beta_gc)
    lag_gc = (per_gc - 1) / (2 * N_gc)

    def f_filt9x_gc(alpha, s, i):
        x = 1 - alpha
        m2 = 36 if i == 9 else 28 if i == 8 else 21 if i == 7 else 15 if i == 6 else 10 if i == 5 else 6 if i == 4 else 3 if i == 3 else 1 if i == 2 else 0
        m3 = 84 if i == 9 else 56 if i == 8 else 35 if i == 7 else 20 if i == 6 else 10 if i == 5 else 4 if i == 4 else 1 if i == 3 else 0
        m4 = 126 if i == 9 else 70 if i == 8 else 35 if i == 7 else 15 if i == 6 else 5 if i == 5 else 1 if i == 4 else 0
        m5 = 126 if i == 9 else 56 if i == 8 else 21 if i == 7 else 6 if i == 6 else 1 if i == 5 else 0
        m6 = 84 if i == 9 else 28 if i == 8 else 7 if i == 7 else 1 if i == 6 else 0
        m7 = 36 if i == 9 else 8 if i == 8 else 1 if i == 7 else 0
        m8 = 9 if i == 9 else 1 if i == 8 else 0
        m9 = 1 if i == 9 else 0

        s = np.where(np.isnan(s), 0, s)
        n = len(s)
        f = np.zeros(n)
        for idx in range(i, n):
            s_val = s[idx]
            f1 = f[idx-1] if idx >= 1 else 0
            f2 = f[idx-2] if idx >= 2 else 0
            f3 = f[idx-3] if idx >= 3 else 0
            f4 = f[idx-4] if idx >= 4 else 0
            f5 = f[idx-5] if idx >= 5 else 0
            f6 = f[idx-6] if idx >= 6 else 0
            f7 = f[idx-7] if idx >= 7 else 0
            f8 = f[idx-8] if idx >= 8 else 0
            f9 = f[idx-9] if idx >= 9 else 0

            term = np.power(alpha, i) * s_val + i * x * f1
            if i >= 2: term = term - m2 * np.power(x, 2) * f2
            if i >= 3: term = term + m3 * np.power(x, 3) * f3
            if i >= 4: term = term - m4 * np.power(x, 4) * f4
            if i >= 5: term = term + m5 * np.power(x, 5) * f5
            if i >= 6: term = term - m6 * np.power(x, 6) * f6
            if i >= 7: term = term + m7 * np.power(x, 7) * f7
            if i >= 8: term = term - m8 * np.power(x, 8) * f8
            if i == 9: term = term + m9 * np.power(x, 9) * f9
            f[idx] = term
        return f

    def f_pole_gc(alpha, srcdata, i):
        f1 = f_filt9x_gc(alpha, srcdata, 1)
        f2 = f_filt9x_gc(alpha, srcdata, 2) if i >= 2 else np.zeros(len(srcdata))
        f3 = f_filt9x_gc(alpha, srcdata, 3) if i >= 3 else np.zeros(len(srcdata))
        f4 = f_filt9x_gc(alpha, srcdata, 4) if i >= 4 else np.zeros(len(srcdata))
        f5 = f_filt9x_gc(alpha, srcdata, 5) if i >= 5 else np.zeros(len(srcdata))
        f6 = f_filt9x_gc(alpha, srcdata, 6) if i >= 6 else np.zeros(len(srcdata))
        f7 = f_filt9x_gc(alpha, srcdata, 7) if i >= 7 else np.zeros(len(srcdata))
        f8 = f_filt9x_gc(alpha, srcdata, 8) if i >= 8 else np.zeros(len(srcdata))
        f9 = f_filt9x_gc(alpha, srcdata, 9) if i == 9 else np.zeros(len(srcdata))

        fn = np.zeros(len(srcdata))
        for idx in range(len(srcdata)):
            if i == 1: fn[idx] = f1[idx]
            elif i == 2: fn[idx] = f2[idx]
            elif i == 3: fn[idx] = f3[idx]
            elif i == 4: fn[idx] = f4[idx]
            elif i == 5: fn[idx] = f5[idx]
            elif i == 6: fn[idx] = f6[idx]
            elif i == 7: fn[idx] = f7[idx]
            elif i == 8: fn[idx] = f8[idx]
            elif i == 9: fn[idx] = f9[idx]

        return fn, f1

    srcdata_gc = src_gc if not modeLag_gc else src_gc + src_gc - src_gc.shift(int(lag_gc))
    trdata_gc = (high - low) if not modeLag_gc else (high - low) + (high - low) - (high - low).shift(int(lag_gc))

    filtn_gc, filt1_gc = f_pole_gc(alpha_gc, srcdata_gc.values, N_gc)
    filtntr_gc, filt1tr_gc = f_pole_gc(alpha_gc, trdata_gc.values, N_gc)

    filt_gc = (filtn_gc + filt1_gc) / 2 if modeFast_gc else filtn_gc
    filttr_gc = (filtntr_gc + filt1tr_gc) / 2 if modeFast_gc else filtntr_gc

    filt_gc = pd.Series(filt_gc, index=df.index)
    filttr_gc = pd.Series(filttr_gc, index=df.index)

    hband_gc = filt_gc + filttr_gc * mult_gc
    lband_gc = filt_gc - filttr_gc * mult_gc

    GCSignals = pd.Series(np.where(filt_gc > filt_gc.shift(1), 1, -1), index=df.index)

    basicLongCondition_GC = (GCSignals > 0) & (close > filt_gc)
    basicShortCondition_GC = (GCSignals < 0) & (close < filt_gc)

    GCSignalsLong = basicLongCondition_GC
    GCSignalsShort = basicShortCondition_GC

    GCSignalsLongCross = (~GCSignalsLong.shift(1).fillna(False).astype(bool)) & GCSignalsLong
    GCSignalsShortCross = (~GCSignalsShort.shift(1).fillna(False).astype(bool)) & GCSignalsShort

    GCSignalsLongFinal = GCSignalsLongCross
    GCSignalsShortFinal = GCSignalsShortCross

    # ============ VFI LOGIC ============
    vfi_length = 130
    vfi_coef = 0.2
    vfi_volCutoff = 2.5
    vfi_smoothLen = 3
    vfi_smoothType = 'EMA'
    vfi_maLength = 30
    vfi_maType = 'SMA'

    avg = hlc3
    inter = np.log(avg) - np.log(avg.shift(1))
    vInter = inter.rolling(30).std()
    cutOff = vfi_coef * vInter * close
    vAve = volume.shift(1).rolling(vfi_length).mean()
    vMax = vAve * vfi_volCutoff
    vC = np.minimum(volume, vMax)
    mf = avg - avg.shift(1)
    vCp = pd.Series(np.where(mf < -cutOff, -vC, 0), index=df.index)
    vCp = pd.Series(np.where(mf > cutOff, vC, vCp), index=df.index)
    sVfi = vCp.rolling(vfi_length).sum() / vAve

    if vfi_smoothType == 'EMA':
        value_vfi = sVfi.ewm(span=vfi_smoothLen, adjust=False).mean()
    elif vfi_smoothType == 'SMA':
        value_vfi = sVfi.rolling(vfi_smoothLen).mean()
    elif vfi_smoothType == 'WMA':
        value_vfi = sVfi.rolling(vfi_smoothLen).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
    else:
        value_vfi = sVfi.ewm(span=vfi_smoothLen, adjust=False).mean()

    if vfi_maType == 'EMA':
        value_ma = value_vfi.ewm(span=vfi_maLength, adjust=False).mean()
    elif vfi_maType == 'SMA':
        value_ma = value_vfi.rolling(vfi_maLength).mean()
    elif vfi_maType == 'WMA':
        value_ma = value_ma = value_vfi.rolling(vfi_maLength).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
    else:
        value_ma = value_vfi.ewm(span=vfi_maLength, adjust=False).mean()

    vfiBullishCondition = (value_vfi > value_ma) & (value_vfi > 0)
    vfiBearishCondition = (value_vfi < value_ma) & (value_vfi < 0)

    # ============ FINAL ENTRY CONDITIONS ============
    bullishEntryCondition = GlitchSignalsLongFinal & GCSignalsLongFinal & vfiBullishCondition
    bearishEntryCondition = GlitchSignalsShortFinal & GCSignalsShortFinal & vfiBearishCondition

    for i in range(len(df)):
        if pd.isna(value_vfi.iloc[i]) or pd.isna(value_ma.iloc[i]) or pd.isna(filt_gc.iloc[i]):
            continue

        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = close.iloc[i]

        if bullishEntryCondition.iloc[i]:
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

        if bearishEntryCondition.iloc[i]:
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