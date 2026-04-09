import pandas as pd
import numpy as np
from datetime import datetime, timezone

def f_filt9x_gc(a, s, i):
    m_weights = {
        2: [0, 1, 3, 6, 10, 15, 21, 28, 36],
        3: [0, 1, 1, 4, 10, 20, 35, 56, 84],
        4: [0, 0, 1, 1, 5, 15, 35, 70, 126],
        5: [0, 0, 0, 1, 1, 6, 21, 56, 126],
        6: [0, 0, 0, 0, 1, 1, 7, 28, 84],
        7: [0, 0, 0, 0, 0, 1, 1, 8, 36],
        8: [0, 0, 0, 0, 0, 0, 1, 1, 9],
        9: [0, 0, 0, 0, 0, 0, 0, 1, 1]
    }
    x = 1 - a
    n = len(s)
    f = np.zeros(n)
    for idx in range(n):
        if idx == 0:
            f[idx] = (a ** i) * s[idx]
        else:
            prev_vals = [f[idx - j] if idx - j >= 0 else 0 for j in range(1, min(i + 1, idx + 1))]
            poly_term = 0
            for k in range(2, i + 1):
                m_idx = k - 2
                if idx - k >= 0:
                    poly_term += m_weights[k][i - 1] * (x ** k) * f[idx - k]
                if k % 2 == 0:
                    poly_term = -poly_term
            f[idx] = (a ** i) * s[idx] + i * x * (f[idx - 1] if idx - 1 >= 0 else 0) - poly_term
    return f

def f_pole_gc(a, s, i):
    f1 = f_filt9x_gc(a, s, 1)
    f_vals = [f1]
    for k in range(2, 10):
        if k <= i:
            f_vals.append(f_filt9x_gc(a, s, k))
        else:
            f_vals.append(np.zeros(len(s)))
    fn = f_vals[i - 1] if 1 <= i <= 9 else np.zeros(len(s))
    return fn, f1

def generate_entries(df: pd.DataFrame) -> list:
    src_gc = (df['high'] + df['low'] + df['close']) / 3.0
    N_gc = 4
    per_gc = 144
    mult_gc = 1.414
    modeLag_gc = False
    modeFast_gc = False
    useGC = True
    crossGC = True
    inverseGC = False
    highlightMovementsGC = True
    
    beta_gc = (1 - np.cos(4 * np.arcsin(1) / per_gc)) / ((1.414 ** (2 / N_gc)) - 1)
    alpha_gc = -beta_gc + np.sqrt(beta_gc ** 2 + 2 * beta_gc)
    lag_gc = (per_gc - 1) / (2 * N_gc)
    
    srcdata_gc = src_gc.values if not modeLag_gc else src_gc.values + src_gc.values - np.roll(src_gc.values, int(lag_gc))
    trdata_gc = (df['high'] - df['low']).values if not modeLag_gc else 2 * (df['high'] - df['low']).values - np.roll((df['high'] - df['low']).values, int(lag_gc))
    
    filtn_gc, filt1_gc = f_pole_gc(alpha_gc, srcdata_gc, N_gc)
    filtntr_gc, filt1tr_gc = f_pole_gc(alpha_gc, trdata_gc, N_gc)
    
    filt_gc = (filtn_gc + filt1_gc) / 2 if modeFast_gc else filtn_gc
    filttr_gc = (filtntr_gc + filt1tr_gc) / 2 if modeFast_gc else filtntr_gc
    
    filt_gc_series = pd.Series(filt_gc)
    close_series = df['close']
    
    GCSignals = pd.Series(0, index=df.index)
    GCSignals[filt_gc > np.roll(filt_gc, 1)] = 1
    GCSignals[filt_gc < np.roll(filt_gc, 1)] = -1
    GCSignals.iloc[0] = 0
    
    basicLongCondition_GC = (GCSignals > 0) & (close_series > filt_gc_series)
    basicShortCondition_GC = (GCSignals < 0) & (close_series < filt_gc_series)
    
    GCSignalsLong = basicLongCondition_GC if useGC else pd.Series(True, index=df.index)
    GCSignalsShort = basicShortCondition_GC if useGC else pd.Series(True, index=df.index)
    
    GCSignalsLongCross = (~GCSignalsLong.shift(1).fillna(False) & GCSignalsLong) if crossGC else GCSignalsLong
    GCSignalsShortCross = (~GCSignalsShort.shift(1).fillna(False) & GCSignalsShort) if crossGC else GCSignalsShort
    
    GCSignalsLongFinal = GCSignalsShortCross if inverseGC else GCSignalsLongCross
    GCSignalsShortFinal = GCSignalsLongCross if inverseGC else GCSignalsShortCross
    
    long_condition = basicLongCondition_GC
    short_condition = basicShortCondition_GC
    
    entries = []
    trade_num = 1
    position_open = False
    
    for i in range(len(df)):
        if not position_open:
            if long_condition.iloc[i]:
                ts = int(df['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': df['close'].iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': df['close'].iloc[i],
                    'raw_price_b': df['close'].iloc[i]
                })
                trade_num += 1
                position_open = True
            elif short_condition.iloc[i]:
                ts = int(df['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': df['close'].iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': df['close'].iloc[i],
                    'raw_price_b': df['close'].iloc[i]
                })
                trade_num += 1
                position_open = True
        else:
            position_open = False
    
    return entries