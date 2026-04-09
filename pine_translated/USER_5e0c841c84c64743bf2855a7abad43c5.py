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

    ts_vals = df['time'].values
    open_vals = df['open'].values
    high_vals = df['high'].values
    low_vals = df['low'].values
    close_vals = df['close'].values
    volume_vals = df['volume'].values
    hl2_vals = (high_vals + low_vals) / 2.0
    hlc3_vals = (high_vals + low_vals + close_vals) / 3.0

    # Glitch parameters
    MaPeriod_glitch = 30
    price_glitch = hl2_vals

    # Glitch long/short
    sma_glitch = pd.Series(price_glitch).rolling(MaPeriod_glitch).mean().values
    longEntry_Glitch = price_glitch > sma_glitch
    shortEntry_Glitch = price_glitch < sma_glitch
    GlitchSignalsLongFinal = longEntry_Glitch
    GlitchSignalsShortFinal = shortEntry_Glitch

    # G-Channel parameters
    src_gc = hlc3_vals
    N_gc = 4
    per_gc = 144
    mult_gc = 1.414
    modeLag_gc = False
    modeFast_gc = False
    useGC = True
    crossGC = True
    inverseGC = False

    beta_gc = (1 - np.cos(4 * np.arcsin(1) / per_gc)) / (np.power(1.414, 2 / N_gc) - 1)
    alpha_gc = -beta_gc + np.sqrt(np.power(beta_gc, 2) + 2 * beta_gc)
    lag_gc = (per_gc - 1) / (2 * N_gc)
    lag_idx = int(lag_gc)

    def f_filt9x_gc(a, s, i):
        x = 1 - a
        m2 = 36 if i == 9 else 28 if i == 8 else 21 if i == 7 else 15 if i == 6 else 10 if i == 5 else 6 if i == 4 else 3 if i == 3 else 1 if i == 2 else 0
        m3 = 84 if i == 9 else 56 if i == 8 else 35 if i == 7 else 20 if i == 6 else 10 if i == 5 else 4 if i == 4 else 1 if i == 3 else 0
        m4 = 126 if i == 9 else 70 if i == 8 else 35 if i == 7 else 15 if i == 6 else 5 if i == 5 else 1 if i == 4 else 0
        m5 = 126 if i == 9 else 56 if i == 8 else 21 if i == 7 else 6 if i == 6 else 1 if i == 5 else 0
        m6 = 84 if i == 9 else 28 if i == 8 else 7 if i == 7 else 1 if i == 6 else 0
        m7 = 36 if i == 9 else 8 if i == 8 else 1 if i == 7 else 0
        m8 = 9 if i == 9 else 1 if i == 8 else 0
        m9 = 1 if i == 9 else 0
        f = np.power(a, i) * s
        f = f + i * x * np.roll(f, 1) - (m2 * np.power(x, 2) * np.roll(f, 2) if i >= 2 else 0)
        f = f + (m3 * np.power(x, 3) * np.roll(f, 3) if i >= 3 else 0)
        f = f - (m4 * np.power(x, 4) * np.roll(f, 4) if i >= 4 else 0)
        f = f + (m5 * np.power(x, 5) * np.roll(f, 5) if i >= 5 else 0)
        f = f - (m6 * np.power(x, 6) * np.roll(f, 6) if i >= 6 else 0)
        f = f + (m7 * np.power(x, 7) * np.roll(f, 7) if i >= 7 else 0)
        f = f - (m8 * np.power(x, 8) * np.roll(f, 8) if i >= 8 else 0)
        f = f + (m9 * np.power(x, 9) * np.roll(f, 9) if i == 9 else 0)
        f[0] = s[0]
        return f

    def f_pole_gc(a, s, i):
        f1 = f_filt9x_gc(a, s, 1)
        f2 = f_filt9x_gc(a, s, 2) if i >= 2 else np.zeros_like(s)
        f3 = f_filt9x_gc(a, s, 3) if i >= 3 else np.zeros_like(s)
        f4 = f_filt9x_gc(a, s, 4) if i >= 4 else np.zeros_like(s)
        f5 = f_filt9x_gc(a, s, 5) if i >= 5 else np.zeros_like(s)
        f6 = f_filt9x_gc(a, s, 6) if i >= 6 else np.zeros_like(s)
        f7 = f_filt9x_gc(a, s, 7) if i >= 7 else np.zeros_like(s)
        f8 = f_filt9x_gc(a, s, 8) if i >= 8 else np.zeros_like(s)
        f9 = f_filt9x_gc(a, s, 9) if i == 9 else np.zeros_like(s)
        fn = f1 if i == 1 else f2 if i == 2 else f3 if i == 3 else f4 if i == 4 else f5 if i == 5 else f6 if i == 6 else f7 if i == 7 else f8 if i == 8 else f9 if i == 9 else np.zeros_like(s)
        return fn, f1

    filt_gc = np.zeros_like(src_gc)
    filttr_gc = np.zeros_like(src_gc)

    for i in range(len(df)):
        srcdata = src_gc[i] if not modeLag_gc else (2 * src_gc[i] - src_gc[i - lag_idx] if i >= lag_idx else src_gc[i])
        trdata = high_vals[i] - low_vals[i]
        if modeLag_gc and i >= lag_idx:
            tr1 = high_vals[i] - low_vals[i]
            tr2 = high_vals[i - lag_idx] - low_vals[i - lag_idx]
            trdata = 2 * tr1 - tr2

        filt_n, filt_1 = f_pole_gc(alpha_gc, srcdata, N_gc)
        filt_n_tr, filt_1_tr = f_pole_gc(alpha_gc, trdata, N_gc)

        if modeFast_gc:
            filt_gc[i] = (filt_n + filt_1) / 2
            filttr_gc[i] = (filt_n_tr + filt_1_tr) / 2
        else:
            filt_gc[i] = filt_n
            filttr_gc[i] = filt_n_tr

    filt_gc = np.array(filt_gc)
    filttr_gc = np.array(filttr_gc)

    GCSignals = np.where(filt_gc > np.roll(filt_gc, 1), 1, np.where(filt_gc < np.roll(filt_gc, 1), -1, 0))
    GCSignals[0] = 0

    basicLongCondition_GC = (GCSignals > 0) & (close_vals > filt_gc)
    basicShortCondition_GC = (GCSignals < 0) & (close_vals < filt_gc)

    GCSignalsLong = basicLongCondition_GC if useGC else np.ones(len(df), dtype=bool)
    GCSignalsShort = basicShortCondition_GC if useGC else np.ones(len(df), dtype=bool)

    GCSignalsLongCross = np.zeros(len(df), dtype=bool)
    GCSignalsShortCross = np.zeros(len(df), dtype=bool)

    for i in range(1, len(df)):
        if crossGC:
            GCSignalsLongCross[i] = not GCSignalsLong[i-1] and GCSignalsLong[i]
            GCSignalsShortCross[i] = not GCSignalsShort[i-1] and GCSignalsShort[i]
        else:
            GCSignalsLongCross[i] = GCSignalsLong[i]
            GCSignalsShortCross[i] = GCSignalsShort[i]

    GCSignalsLongFinal = GCSignalsShortCross if inverseGC else GCSignalsLongCross
    GCSignalsShortFinal = GCSignalsLongCross if inverseGC else GCSignalsShortCross

    # VFI parameters
    vfi_length = 130
    vfi_coef = 0.2
    vfi_volCutoff = 2.5
    vfi_smoothLen = 3
    vfi_smoothType = 'EMA'
    vfi_maLength = 30
    vfi_maType = 'SMA'

    inter = np.zeros(len(df))
    for i in range(1, len(df)):
        if hlc3_vals[i-1] > 0:
            inter[i] = np.log(hlc3_vals[i]) - np.log(hlc3_vals[i-1])

    vInter = pd.Series(inter).rolling(30).std().fillna(0).values
    cutOff = vfi_coef * vInter * close_vals

    vAve = pd.Series(volume_vals).shift(1).rolling(vfi_length).mean().fillna(0).values
    vMax = vAve * vfi_volCutoff
    vC = np.minimum(volume_vals, vMax)

    mf = np.zeros(len(df))
    for i in range(1, len(df)):
        mf[i] = hlc3_vals[i] - hlc3_vals[i-1]

    vCp = np.where(mf > cutOff, vC, np.where(mf < -cutOff, -vC, 0))
    sVfi = pd.Series(vCp).rolling(vfi_length).sum() / vAve
    sVfi = sVfi.fillna(0)

    if vfi_smoothType == 'EMA':
        value_vfi = pd.Series(sVfi).ewm(span=vfi_smoothLen, adjust=False).mean().values
    elif vfi_smoothType == 'SMA':
        value_vfi = pd.Series(sVfi).rolling(vfi_smoothLen).mean().values
    elif vfi_smoothType == 'WMA':
        value_vfi = pd.Series(sVfi).rolling(vfi_smoothLen).apply(lambda x: np.dot(x, np.arange(1, vfi_smoothLen+1))/np.sum(np.arange(1, vfi_smoothLen+1)), raw=True).values
    else:
        value_vfi = pd.Series(sVfi).ewm(span=vfi_smoothLen, adjust=False).mean().values

    if vfi_maType == 'EMA':
        value_ma = pd.Series(value_vfi).ewm(span=vfi_maLength, adjust=False).mean().values
    elif vfi_maType == 'SMA':
        value_ma = pd.Series(value_vfi).rolling(vfi_maLength).mean().values
    elif vfi_maType == 'WMA':
        value_ma = pd.Series(value_vfi).rolling(vfi_maLength).apply(lambda x: np.dot(x, np.arange(1, vfi_maLength+1))/np.sum(np.arange(1, vfi_maLength+1)), raw=True).values
    else:
        value_ma = pd.Series(value_vfi).rolling(vfi_maLength).mean().values

    vfiBullishCondition = (value_vfi > value_ma) & (value_vfi > 0)

    # Entry conditions
    long_entry = GlitchSignalsLongFinal & GCSignalsLongFinal & vfiBullishCondition
    short_entry = GlitchSignalsShortFinal & GCSignalsShortFinal

    trade_num = 1
    for i in range(1, len(df)):
        if long_entry[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts_vals[i]),
                'entry_time': datetime.fromtimestamp(ts_vals[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close_vals[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_vals[i]),
                'raw_price_b': float(close_vals[i])
            })
            trade_num += 1
        elif short_entry[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts_vals[i]),
                'entry_time': datetime.fromtimestamp(ts_vals[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close_vals[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_vals[i]),
                'raw_price_b': float(close_vals[i])
            })
            trade_num += 1

    return entries