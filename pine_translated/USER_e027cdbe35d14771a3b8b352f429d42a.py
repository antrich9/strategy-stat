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
    
    # ── INPUTS ──────────────────────────────────────────────────────────────
    cci_len = 25
    trail_len = 20
    upper_thresh = 50
    lower_thresh = -50
    rf_type = "Type 1"
    rf_mov_src = "Close"
    rf_qty = 2.618
    rf_scale = "Average Change"
    rf_per = 14
    rf_smooth = True
    rf_smooth_per = 27
    useTDFI = True
    crossTDFI = True
    inverseTDFI = False
    lookbackTDFI = 13
    mmaLengthTDFI = 13
    mmaModeTDFI = "ema"
    smmaLengthTDFI = 13
    smmaModeTDFI = "ema"
    nLengthTDFI = 3
    filterHighTDFI = 0.05
    filterLowTDFI = -0.05
    adx_len = 14
    adx_ma_len = 14
    adx_threshold = 0.0
    
    # ── HELPER: Wilder EMA (Cond_EMA) ───────────────────────────────────────
    def cond_ema_series(series, n):
        result = np.full(len(series), np.nan)
        ema_val = np.nan
        alpha = 2.0 / (n + 1)
        for i in range(len(series)):
            if not np.isnan(series.iloc[i]):
                if np.isnan(ema_val):
                    ema_val = series.iloc[i]
                else:
                    ema_val = (series.iloc[i] - ema_val) * alpha + ema_val
            result[i] = ema_val
        return pd.Series(result, index=series.index)
    
    # ── CCI ─────────────────────────────────────────────────────────────────
    typical = (df['high'] + df['low'] + df['close']) / 3.0
    sma_typical = typical.rolling(cci_len).mean()
    mad = typical.rolling(cci_len).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (typical - sma_typical) / (0.015 * mad + 1e-10)
    
    # ── CTR (Commodity Trend Reactor) ───────────────────────────────────────
    low_ = df['low'].rolling(trail_len).min()
    high_ = df['high'].rolling(trail_len).max()
    
    ctr_trend = pd.Series(False, index=df.index)
    ctr_trail = pd.Series(np.nan, index=df.index)
    
    # Calculate CTR trend with crossover/crossunder
    for i in range(1, len(df)):
        prev_cci = cci.iloc[i-1]
        curr_cci = cci.iloc[i]
        
        # Crossover: prev <= upper_thresh and curr > upper_thresh
        if prev_cci <= upper_thresh and curr_cci > upper_thresh:
            ctr_trend.iloc[i] = True
        # Crossunder: prev >= lower_thresh and curr < lower_thresh
        elif prev_cci >= lower_thresh and curr_cci < lower_thresh:
            ctr_trend.iloc[i] = False
        else:
            ctr_trend.iloc[i] = ctr_trend.iloc[i-1]
    
    ctr_trail = np.where(ctr_trend, low_, high_)
    
    # ── RANGE FILTER ────────────────────────────────────────────────────────
    h_val = df['high'] if rf_mov_src == "Wicks" else df['close']
    l_val = df['low'] if rf_mov_src == "Wicks" else df['close']
    
    # AC (Average Change)
    ac = np.abs(df['close'] - df['close'].shift(1))
    ac_ema = cond_ema_series(ac, rf_per)
    rng_base = rf_qty * ac_ema
    
    # ATR (Wilder)
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(np.abs(df['high'] - df['close'].shift(1)),
                               np.abs(df['low'] - df['close'].shift(1))))
    atr_rf = cond_ema_series(tr, rf_per)
    
    if rf_scale == "ATR":
        rng_ = rf_qty * atr_rf
    elif rf_scale == "Average Change":
        rng_ = rf_qty * ac_ema
    else:
        rng_ = rng_base
    
    if rf_smooth:
        rng_smooth = cond_ema_series(rng_, rf_smooth_per)
        r = rng_smooth
    else:
        r = rng_
    
    # Range Filter logic
    filt = (h_val + l_val) / 2.0
    filt_arr = filt.values.copy().astype(float)
    r_arr = r.values
    
    for i in range(1, len(df)):
        if rf_type == "Type 1":
            h_r = h_val.iloc[i] - r_arr[i]
            l_r = l_val.iloc[i] + r_arr[i]
            if h_r > filt_arr[i-1]:
                filt_arr[i] = h_r
            elif l_r < filt_arr[i-1]:
                filt_arr[i] = l_r
            else:
                filt_arr[i] = filt_arr[i-1]
        else:
            if h_val.iloc[i] >= filt_arr[i-1] + r_arr[i]:
                filt_arr[i] = filt_arr[i-1] + np.floor(np.abs(h_val.iloc[i] - filt_arr[i-1]) / r_arr[i]) * r_arr[i]
            elif l_val.iloc[i] <= filt_arr[i-1] - r_arr[i]:
                filt_arr[i] = filt_arr[i-1] - np.floor(np.abs(l_val.iloc[i] - filt_arr[i-1]) / r_arr[i]) * r_arr[i]
            else:
                filt_arr[i] = filt_arr[i-1]
    
    rf_filt = pd.Series(filt_arr, index=df.index)
    rf_dir = pd.Series(0.0, index=df.index)
    
    for i in range(1, len(df)):
        if rf_filt.iloc[i] > rf_filt.iloc[i-1]:
            rf_dir.iloc[i] = 1
        elif rf_filt.iloc[i] < rf_filt.iloc[i-1]:
            rf_dir.iloc[i] = -1
        else:
            rf_dir.iloc[i] = rf_dir.iloc[i-1]
    
    rf_bullish = rf_dir == 1
    rf_bearish = rf_dir == -1
    
    # ── TDFI ─────────────────────────────────────────────────────────────────
    price_tdfi = df['close'] * 1000
    
    def tema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3
    
    def ma_tdfi(mode, src, length):
        if mode == "ema":
            return src.ewm(span=length, adjust=False).mean()
        elif mode == "wma":
            weights = np.arange(1, length + 1)
            return src.rolling(length).apply(lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum() if len(x) == length else np.nan, raw=True)
        elif mode == "swma":
            return (src.shift(2) + 2 * src.shift(1) + 2 * src + 2 * src.shift(-1) + src.shift(-2)) / 6
        elif mode == "vwma":
            return (src * df['volume']).rolling(length).mean() / df['volume'].rolling(length).mean()
        elif mode == "hull":
            half_len = int(length / 2)
            sqrt_len = int(np.sqrt(length))
            hull = 2 * src.rolling(half_len).mean() - src.rolling(length).mean()
            return hull.rolling(sqrt_len).mean()
        elif mode == "tema":
            return tema(src, length)
        else:
            return src.rolling(length).mean()
    
    mma_tdfi = ma_tdfi(mmaModeTDFI, price_tdfi, mmaLengthTDFI)
    smma_tdfi = ma_tdfi(smmaModeTDFI, mma_tdfi, smmaLengthTDFI)
    
    impetmma_tdfi = mma_tdfi - mma_tdfi.shift(1)
    impetsmma_tdfi = smma_tdfi - smma_tdfi.shift(1)
    divma_tdfi = np.abs(mma_tdfi - smma_tdfi)
    averimpet_tdfi = (impetmma_tdfi + impetsmma_tdfi) / 2
    
    lookback_full = lookbackTDFI * nLengthTDFI
    tdf_tdfi = np.power(divma_tdfi, 1) * np.power(averimpet_tdfi, nLengthTDFI)
    highest_tdf = tdf_tdfi.abs().rolling(lookback_full).max()
    signal_tdfi = tdf_tdfi / (highest_tdf + 1e-10)
    
    signal_long_tdfi = (signal_tdfi > filterHighTDFI) if not crossTDFI else ((signal_tdfi > filterHighTDFI) & (signal_tdfi.shift(1) <= filterHighTDFI))
    signal_short_tdfi = (signal_tdfi < filterLowTDFI) if not crossTDFI else ((signal_tdfi < filterLowTDFI) & (signal_tdfi.shift(1) >= filterLowTDFI))
    
    if not crossTDFI:
        signal_long_tdfi = signal_tdfi > filterHighTDFI
        signal_short_tdfi = signal_tdfi < filterLowTDFI
    
    if inverseTDFI:
        final_long_tdfi = signal_short_tdfi
        final_short_tdfi = signal_long_tdfi
    else:
        final_long_tdfi = signal_long_tdfi
        final_short_tdfi = signal_short_tdfi
    
    if not useTDFI:
        final_long_tdfi = True
        final_short_tdfi = True
    
    # ── ADX ──────────────────────────────────────────────────────────────────
    high_diff = df['high'] - df['high'].shift(1)
    low_diff = df['low'].shift(1) - df['low']
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    plus_dm_series = pd.Series(plus_dm, index=df.index)
    minus_dm_series = pd.Series(minus_dm, index=df.index)
    
    plus_dm_ema = cond_ema_series(plus_dm_series, adx_len)
    minus_dm_ema = cond_ema_series(minus_dm_series, adx_len)
    atr_adx = cond_ema_series(tr, adx_len)
    
    di_plus = 100 * plus_dm_ema / (atr_adx + 1e-10)
    di_minus = 100 * minus_dm_ema / (atr_adx + 1e-10)
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_val = dx.ewm(span=adx_len, adjust=False).mean()
    
    adx_ma = adx_val.rolling(adx_ma_len).mean()
    adx_filter = adx_val > (adx_ma + adx_threshold)
    
    # ── ENTRY CONDITIONS ────────────────────────────────────────────────────
    long_condition = (ctr_trend == True) & (df['close'] > pd.Series(ctr_trail, index=df.index)) & rf_bullish & final_long_tdfi & adx_filter
    short_condition = (ctr_trend == False) & (df['close'] < pd.Series(ctr_trail, index=df.index)) & rf_bearish & final_short_tdfi & adx_filter
    
    # ── LATCH ENTRY TRIGGERS & GENERATE ENTRIES ─────────────────────────────
    entries = []
    trade_num = 1
    position_open = False
    long_triggered = False
    short_triggered = False
    
    for i in range(1, len(df)):
        # Latch triggers
        if long_condition.iloc[i]:
            long_triggered = True
            short_triggered = False
        
        if short_condition.iloc[i]:
            short_triggered = True
            long_triggered = False
        
        # Check for entry (only when flat)
        if not position_open:
            if long_triggered:
                ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': df['close'].iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': df['close'].iloc[i],
                    'raw_price_b': df['close'].iloc[i]
                })
                trade_num += 1
                position_open = True
                long_triggered = False
                
            elif short_triggered:
                ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': df['close'].iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': df['close'].iloc[i],
                    'raw_price_b': df['close'].iloc[i]
                })
                trade_num += 1
                position_open = True
                short_triggered = False
    
    return entries