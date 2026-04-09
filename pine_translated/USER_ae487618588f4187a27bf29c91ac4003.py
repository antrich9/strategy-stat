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
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']
    time = df['time']

    # Use HLC3 as source
    hlc3 = (high + low + close) / 3

    # G-Channel Parameters
    src_gc = hlc3
    N_gc = 4
    per_gc = 144
    mult_gc = 1.414
    useGC = True
    crossGC = True
    highlightMovementsGC = True
    inverseGC = False
    modeLag_gc = False
    modeFast_gc = False

    # G-Channel Beta and Alpha
    beta_gc = (1 - np.cos(4 * np.arcsin(1) / per_gc)) / (np.power(1.414, 2 / N_gc) - 1)
    alpha_gc = -beta_gc + np.sqrt(np.power(beta_gc, 2) + 2 * beta_gc)

    # Lag
    lag_gc = (per_gc - 1) / (2 * N_gc)

    # Data
    srcdata_gc = src_gc if not modeLag_gc else src_gc + src_gc - pd.Series(src_gc).shift(int(lag_gc)).fillna(src_gc)
    tr_gc = high - low
    trdata_gc = tr_gc if not modeLag_gc else tr_gc + tr_gc - pd.Series(tr_gc).shift(int(lag_gc)).fillna(tr_gc)

    # Filter function f_filt9x_gc implementation
    def f_filt9x_gc_series(alpha, source, poles):
        result = pd.Series(index=source.index, dtype=float)
        m2_vals = {9: 36, 8: 28, 7: 21, 6: 15, 5: 10, 4: 6, 3: 3, 2: 1}
        m3_vals = {9: 84, 8: 56, 7: 35, 6: 20, 5: 10, 4: 4, 3: 1}
        m4_vals = {9: 126, 8: 70, 7: 35, 6: 15, 5: 5, 4: 1}
        m5_vals = {9: 126, 8: 56, 7: 21, 6: 6, 5: 1}
        m6_vals = {9: 84, 8: 28, 7: 7, 6: 1}
        m7_vals = {9: 36, 8: 8, 7: 1}
        m8_vals = {9: 9, 8: 1}
        m9_vals = {9: 1}

        m2 = m2_vals.get(poles, 0)
        m3 = m3_vals.get(poles, 0)
        m4 = m4_vals.get(poles, 0)
        m5 = m5_vals.get(poles, 0)
        m6 = m6_vals.get(poles, 0)
        m7 = m7_vals.get(poles, 0)
        m8 = m8_vals.get(poles, 0)
        m9 = m9_vals.get(poles, 0)

        x = 1 - alpha
        filt_arr = np.zeros(len(source))
        filt_arr_delayed = np.zeros(len(source))

        for i in range(len(source)):
            s_val = source.iloc[i] if not pd.isna(source.iloc[i]) else 0
            f1 = filt_arr[i-1] if i >= 1 else 0
            f2 = filt_arr[i-2] if i >= 2 else 0
            f3 = filt_arr[i-3] if i >= 3 else 0
            f4 = filt_arr[i-4] if i >= 4 else 0
            f5 = filt_arr[i-5] if i >= 5 else 0
            f6 = filt_arr[i-6] if i >= 6 else 0
            f7 = filt_arr[i-7] if i >= 7 else 0
            f8 = filt_arr[i-8] if i >= 8 else 0
            f9 = filt_arr[i-9] if i >= 9 else 0

            term1 = np.power(alpha, poles) * s_val
            term2 = poles * x * f1
            term3 = 0
            if poles >= 2:
                term3 = m2 * np.power(x, 2) * f2
            term4 = 0
            if poles >= 3:
                term4 = m3 * np.power(x, 3) * f3
            term5 = 0
            if poles >= 4:
                term5 = m4 * np.power(x, 4) * f4
            term6 = 0
            if poles >= 5:
                term6 = m5 * np.power(x, 5) * f5
            term7 = 0
            if poles >= 6:
                term7 = m6 * np.power(x, 6) * f6
            term8 = 0
            if poles >= 7:
                term8 = m7 * np.power(x, 7) * f7
            term9 = 0
            if poles >= 8:
                term9 = m8 * np.power(x, 8) * f8
            term10 = 0
            if poles == 9:
                term10 = m9 * np.power(x, 9) * f9

            filt_arr[i] = term1 + term2 - term3 + term4 - term5 + term6 - term7 + term8 - term9 + term10

        result[:] = filt_arr
        return result

    def f_pole_gc_series(alpha, source, poles):
        filt1 = f_filt9x_gc_series(alpha, source, 1)
        filtn = f_filt9x_gc_series(alpha, source, poles)
        return filtn, filt1

    filtn_gc, filt1_gc = f_pole_gc_series(alpha_gc, srcdata_gc, N_gc)
    filtntr_gc, filt1tr_gc = f_pole_gc_series(alpha_gc, trdata_gc, N_gc)

    filt_gc = (filtn_gc + filt1_gc) / 2 if modeFast_gc else filtn_gc
    filttr_gc = (filtntr_gc + filt1tr_gc) / 2 if modeFast_gc else filtntr_gc

    hband_gc = filt_gc + filttr_gc * mult_gc
    lband_gc = filt_gc - filttr_gc * mult_gc

    # G-Channel Signals
    GCSignals = (filt_gc > filt_gc.shift(1)).astype(int).replace(0, -1)
    basicLongCondition_GC = (GCSignals > 0) & (close > filt_gc)
    basicShortCondition_GC = (GCSignals < 0) & (close < filt_gc)

    if useGC:
        GCSignalsLong = basicLongCondition_GC if highlightMovementsGC else (close > filt_gc)
        GCSignalsShort = basicShortCondition_GC if highlightMovementsGC else (close < filt_gc)
    else:
        GCSignalsLong = pd.Series(True, index=close.index)
        GCSignalsShort = pd.Series(True, index=close.index)

    GCSignalsLongCross = GCSignalsLong & (~GCSignalsLong.shift(1).fillna(False)) if crossGC else GCSignalsLong
    GCSignalsShortCross = GCSignalsShort & (~GCSignalsShort.shift(1).fillna(False)) if crossGC else GCSignalsShort

    GCSignalsLongFinal = GCSignalsShortCross if inverseGC else GCSignalsLongCross
    GCSignalsShortFinal = GCSignalsLongCross if inverseGC else GCSignalsShortCross

    # JMA (Jurik-like Moving Average) - Simplified T3 implementation
    jma_length = 14
    jma_phase = 0
    jma_power = 2.0

    # T3 Moving Average (simplified JMA-like)
    ema1 = close.ewm(span=jma_length, adjust=False).mean()
    ema2 = ema1.ewm(span=jma_length, adjust=False).mean()
    ema3 = ema2.ewm(span=jma_length, adjust=False).mean()
    ema4 = ema3.ewm(span=jma_length, adjust=False).mean()
    ema5 = ema4.ewm(span=jma_length, adjust=False).mean()
    ema6 = ema5.ewm(span=jma_length, adjust=False).mean()
    c1 = -3
    c2 = 3
    c3 = -1
    c4 = 1
    c5 = -3
    c6 = 1
    c7 = 1
    jma = c1 * ema6 + c2 * ema5 + c3 * ema4 + c4 * ema3 + c5 * ema2 + c6 * ema1 + c7 * close

    # FluxWave Oscillator
    flux_smooth = 1
    flux_length = 34
    fluxSource = close

    # T3 for FluxWave
    ema1_flux = fluxSource.ewm(span=flux_length, adjust=False).mean()
    ema2_flux = ema1_flux.ewm(span=flux_length, adjust=False).mean()
    ema3_flux = ema2_flux.ewm(span=flux_length, adjust=False).mean()
    ema4_flux = ema3_flux.ewm(span=flux_length, adjust=False).mean()
    ema5_flux = ema4_flux.ewm(span=flux_length, adjust=False).mean()
    ema6_flux = ema5_flux.ewm(span=flux_length, adjust=False).mean()
    fluxwave_raw = 6 * ema1_flux - 15 * ema2_flux + 10 * ema3_flux - 3 * ema4_flux + ema5_flux - ema6_flux
    fluxwave_smooth = fluxwave_raw.ewm(span=flux_smooth, adjust=False).mean() if flux_smooth > 1 else fluxwave_raw

    fluxwave_signal = fluxwave_smooth.ewm(span=5, adjust=False).mean()
    fluxwave_hist = fluxwave_smooth - fluxwave_signal

    fluxwave_long = fluxwave_hist > 0
    fluxwave_short = fluxwave_hist < 0

    # QuickSilver Band System
    qs_period = 14
    qs_mult = 3.0

    # Mesa Adaptive Moving Average (simplified)
    detrend = close - close.rolling(qs_period).mean()
    smooth_detrend = detrend.ewm(span=qs_period, adjust=False).mean()
    qs_mama = close.ewm(span=qs_period, adjust=False).mean()
    qs_fama = qs_mama.ewm(span=qs_period // 2, adjust=False).mean()
    qs_upper = qs_mama + qs_mult * close.rolling(qs_period).std()
    qs_lower = qs_fama - qs_mult * close.rolling(qs_period).std()

    quicksilver_long = close > qs_mama
    quicksilver_short = close < qs_fama

    # Volume Filter - Normalized Volume Energy
    vol_length = 20
    vol_ma = volume.rolling(vol_length).mean()
    vol_std = volume.rolling(vol_length).std()
    vol_upper = vol_ma + vol_std
    vol_lower = vol_ma - vol_std
    volume_filter = volume > vol_ma

    # Entry conditions combination
    use_alma_signal = True
    use_flux_signal = True
    use_qs_signal = True
    use_vol_filter = True

    alma_long = close > jma
    alma_short = close < jma

    # Combined entry conditions
    long_condition = (
        (not use_alma_signal or alma_long) &
        (not use_flux_signal or fluxwave_long) &
        (not use_qs_signal or quicksilver_long) &
        (not use_vol_filter or volume_filter) &
        GCSignalsLongFinal
    )

    short_condition = (
        (not use_alma_signal or alma_short) &
        (not use_flux_signal or fluxwave_short) &
        (not use_qs_signal or quicksilver_short) &
        (not use_vol_filter or volume_filter) &
        GCSignalsShortFinal
    )

    # Align all series to the same length
    min_len = min(len(long_condition), len(short_condition), len(close))
    long_condition = long_condition.iloc[:min_len]
    short_condition = short_condition.iloc[:min_len]
    close_aligned = close.iloc[:min_len]
    time_aligned = time.iloc[:min_len]

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(min_len):
        if pd.isna(long_condition.iloc[i]) or pd.isna(short_condition.iloc[i]):
            continue
        if long_condition.iloc[i]:
            ts = int(time_aligned.iloc[i])
            entry_time_str = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat() if ts > 1e12 else datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': close_aligned.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_aligned.iloc[i],
                'raw_price_b': close_aligned.iloc[i]
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            ts = int(time_aligned.iloc[i])
            entry_time_str = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat() if ts > 1e12 else datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': close_aligned.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_aligned.iloc[i],
                'raw_price_b': close_aligned.iloc[i]
            })
            trade_num += 1

    return entries