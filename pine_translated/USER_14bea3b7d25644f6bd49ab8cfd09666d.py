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
    results = []
    trade_num = 1

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_col = df['open']

    hl2 = (high + low) / 2
    hlc3 = (high + low + close) / 3

    # ATR (Wilder)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()

    # Glitch Logic
    MaPeriod_glitch = 30
    price_glitch = hl2
    sma_glitch = price_glitch.rolling(MaPeriod_glitch).mean()
    longEntry_Glitch = price_glitch > sma_glitch
    shortEntry_Glitch = price_glitch < sma_glitch
    GlitchSignalsLongFinal = longEntry_Glitch
    GlitchSignalsShortFinal = shortEntry_Glitch

    # G-Channel Logic
    N_gc = 4
    per_gc = 144
    mult_gc = 1.414
    modeLag_gc = False
    modeFast_gc = False
    useGC = True
    crossGC = True
    inverseGC = False
    highlightMovementsGC = True

    src_gc = hlc3

    beta_gc = (1 - np.cos(4 * np.arcsin(1) / per_gc)) / (np.power(1.414, 2 / N_gc) - 1)
    alpha_gc = -beta_gc + np.sqrt(np.power(beta_gc, 2) + 2 * beta_gc)

    lag_gc = (per_gc - 1) / (2 * N_gc)

    def f_filt9x_gc(a, s, i):
        m2 = 36 if i == 9 else 28 if i == 8 else 21 if i == 7 else 15 if i == 6 else 10 if i == 5 else 6 if i == 4 else 3 if i == 3 else 1 if i == 2 else 0
        m3 = 84 if i == 9 else 56 if i == 8 else 35 if i == 7 else 20 if i == 6 else 10 if i == 5 else 4 if i == 4 else 1 if i == 3 else 0
        m4 = 126 if i == 9 else 70 if i == 8 else 35 if i == 7 else 15 if i == 6 else 5 if i == 5 else 1 if i == 4 else 0
        m5 = 126 if i == 9 else 56 if i == 8 else 21 if i == 7 else 6 if i == 6 else 1 if i == 5 else 0
        m6 = 84 if i == 9 else 28 if i == 8 else 7 if i == 7 else 1 if i == 6 else 0
        m7 = 36 if i == 9 else 8 if i == 8 else 1 if i == 7 else 0
        m8 = 9 if i == 9 else 1 if i == 8 else 0
        m9 = 1 if i == 9 else 0

        x = 1 - a
        f_vals = [0.0]
        for idx in range(1, len(s) + 1):
            s_val = s.iloc[idx - 1] if idx <= len(s) else 0
            f_prev1 = f_vals[idx - 1] if idx > 0 else 0
            f_prev2 = f_vals[idx - 2] if idx > 1 else 0
            f_prev3 = f_vals[idx - 3] if idx > 2 else 0
            f_prev4 = f_vals[idx - 4] if idx > 3 else 0
            f_prev5 = f_vals[idx - 5] if idx > 4 else 0
            f_prev6 = f_vals[idx - 6] if idx > 5 else 0
            f_prev7 = f_vals[idx - 7] if idx > 6 else 0
            f_prev8 = f_vals[idx - 8] if idx > 7 else 0
            f_prev9 = f_vals[idx - 9] if idx > 8 else 0

            f = np.power(a, i) * s_val + i * x * f_prev1
            if i >= 2:
                f -= m2 * np.power(x, 2) * f_prev2
            if i >= 3:
                f += m3 * np.power(x, 3) * f_prev3
            if i >= 4:
                f -= m4 * np.power(x, 4) * f_prev4
            if i >= 5:
                f += m5 * np.power(x, 5) * f_prev5
            if i >= 6:
                f -= m6 * np.power(x, 6) * f_prev6
            if i >= 7:
                f += m7 * np.power(x, 7) * f_prev7
            if i >= 8:
                f -= m8 * np.power(x, 8) * f_prev8
            if i == 9:
                f += m9 * np.power(x, 9) * f_prev9
            f_vals.append(f)
        return pd.Series(f_vals[1:], index=s.index)

    def f_pole_gc(a, s, i):
        f1 = f_filt9x_gc(a, s, 1)
        f2 = f_filt9x_gc(a, s, 2) if i >= 2 else 0
        f3 = f_filt9x_gc(a, s, 3) if i >= 3 else 0
        f4 = f_filt9x_gc(a, s, 4) if i >= 4 else 0
        f5 = f_filt9x_gc(a, s, 5) if i >= 5 else 0
        f6 = f_filt9x_gc(a, s, 6) if i >= 6 else 0
        f7 = f_filt9x_gc(a, s, 7) if i >= 2 else 0
        f8 = f_filt9x_gc(a, s, 8) if i >= 8 else 0
        f9 = f_filt9x_gc(a, s, 9) if i == 9 else 0

        if i == 1:
            fn = f1
        elif i == 2:
            fn = f2
        elif i == 3:
            fn = f3
        elif i == 4:
            fn = f4
        elif i == 5:
            fn = f5
        elif i == 6:
            fn = f6
        elif i == 7:
            fn = f7
        elif i == 8:
            fn = f8
        elif i == 9:
            fn = f9
        else:
            fn = pd.Series([np.nan] * len(s), index=s.index)
        return fn, f1

    srcdata_gc = src_gc if not modeLag_gc else src_gc + src_gc - src_gc.shift(int(lag_gc))
    trdata_gc = tr if not modeLag_gc else tr + tr - tr.shift(int(lag_gc))

    filtn_gc, filt1_gc = f_pole_gc(alpha_gc, srcdata_gc, N_gc)
    filtntr_gc, filt1tr_gc = f_pole_gc(alpha_gc, trdata_gc, N_gc)

    filt_gc = (filtn_gc + filt1_gc) / 2 if modeFast_gc else filtn_gc
    filttr_gc = (filtntr_gc + filt1tr_gc) / 2 if modeFast_gc else filtntr_gc

    hband_gc = filt_gc + filttr_gc * mult_gc
    lband_gc = filt_gc - filttr_gc * mult_gc

    GCSignals = pd.Series([1 if filt_gc.iloc[i] > filt_gc.shift(1).iloc[i] else -1 for i in range(len(filt_gc))], index=filt_gc.index)

    basicLongCondition_GC = (GCSignals > 0) & (close > filt_gc)
    basicShortCondition_GC = (GCSignals < 0) & (close < filt_gc)

    GCSignalsLong = (close > filt_gc) if not useGC else ((basicLongCondition_GC) if highlightMovementsGC else (close > filt_gc))
    GCSignalsShort = (close < filt_gc) if not useGC else ((basicShortCondition_GC) if highlightMovementsGC else (close < filt_gc))

    GCSignalsLongCross = (~GCSignalsLong.shift(1).fillna(False) & GCSignalsLong) if crossGC else GCSignalsLong
    GCSignalsShortCross = (~GCSignalsShort.shift(1).fillna(False) & GCSignalsShort) if crossGC else GCSignalsShort

    GCSignalsLongFinal = GCSignalsShortCross if inverseGC else GCSignalsLongCross
    GCSignalsShortFinal = GCSignalsLongCross if inverseGC else GCSignalsShortCross

    # VFI Logic
    vfi_length = 130
    vfi_coef = 0.2
    vfi_volCutoff = 2.5
    vfi_smoothLen = 3
    vfi_smoothType = 'EMA'
    vfi_maLength = 30
    vfi_maType = 'SMA'

    avg = hlc3.shift(1).fillna(hlc3)
    inter = np.log(avg) - np.log(avg.shift(1).fillna(avg))
    vInter = inter.rolling(30).std()
    cutOff = vfi_coef * vInter * close
    vAve = volume.shift(1).fillna(volume).rolling(vfi_length).mean()
    vMax = vAve * vfi_volCutoff
    vC = volume.where(volume <= vMax, vMax)
    mf = avg - avg.shift(1).fillna(avg)
    vCp = vC.where(mf > cutOff, -vC.where(mf < -cutOff, 0))
    sVfi = vCp.rolling(vfi_length).sum() / vAve

    if vfi_smoothType == 'SMA':
        value_vfi = sVfi.rolling(vfi_smoothLen).mean()
    elif vfi_smoothType == 'EMA':
        value_vfi = sVfi.ewm(span=vfi_smoothLen, adjust=False).mean()
    elif vfi_smoothType == 'WMA':
        value_vfi = sVfi.rolling(vfi_smoothLen).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
    else:
        value_vfi = sVfi.ewm(span=vfi_smoothLen, adjust=False).mean()

    if vfi_maType == 'SMA':
        value_ma = value_vfi.rolling(vfi_maLength).mean()
    elif vfi_maType == 'EMA':
        value_ma = value_vfi.ewm(span=vfi_maLength, adjust=False).mean()
    elif vfi_maType == 'WMA':
        value_ma = value_vfi.rolling(vfi_maLength).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
    else:
        value_ma = value_vfi.ewm(span=vfi_maLength, adjust=False).mean()

    vfiBullishCondition = (value_vfi > value_ma) & (value_vfi > 0)

    # Combined entry conditions
    longCondition = GlitchSignalsLongFinal & GCSignalsLongFinal & vfiBullishCondition
    shortCondition = GlitchSignalsShortFinal & GCSignalsShortFinal

    # Generate entries
    for i in range(len(df)):
        if pd.isna(GlitchSignalsLongFinal.iloc[i]) or pd.isna(GCSignalsLongFinal.iloc[i]) or pd.isna(value_vfi.iloc[i]):
            continue

        if longCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1

        if shortCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1

    return results