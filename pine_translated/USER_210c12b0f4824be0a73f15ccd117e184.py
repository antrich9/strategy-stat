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
    
    df = df.copy().reset_index(drop=True)
    
    # === 1. COMMODITY TREND REACTOR (CTR) ===
    len_ctr = 25
    t_len_ctr = 20
    upper_ctr = 50
    lower_ctr = -50
    
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    
    def wilder_mean(series, window):
        alpha = 2.0 / (window + 1)
        result = series.rolling(window).mean()
        for i in range(window, len(series)):
            result.iloc[i] = series.iloc[i] * alpha + result.iloc[i-1] * (1 - alpha)
        return result
    
    sma_ctr = typical_price.rolling(len_ctr).mean()
    mean_dev = typical_price.rolling(len_ctr).apply(lambda x: np.abs(x - x.median()).mean(), raw=True)
    cci = (typical_price - sma_ctr) / (0.015 * mean_dev)
    
    lowest_line = df['low'].rolling(t_len_ctr).min()
    highest_line = df['high'].rolling(t_len_ctr).max()
    
    ctr_trend = pd.Series(False, index=df.index)
    for i in range(1, len(df)):
        if cci.iloc[i] > upper_ctr and cci.iloc[i-1] <= upper_ctr:
            ctr_trend.iloc[i] = True
        elif cci.iloc[i] < lower_ctr and cci.iloc[i-1] >= lower_ctr:
            ctr_trend.iloc[i] = False
        else:
            ctr_trend.iloc[i] = ctr_trend.iloc[i-1]
    
    trail_line = np.where(ctr_trend, lowest_line, highest_line)
    ctr_long = ctr_trend & (df['close'] > trail_line)
    ctr_short = (~ctr_trend) & (df['close'] < trail_line)
    
    # === 2. TDFI ===
    mma_length_tdfi = 13
    smma_length_tdfi = 13
    lookback_tdfi = 13
    n_length_tdfi = 3
    filter_high_tdfi = 0.05
    filter_low_tdfi = -0.05
    
    price_tdfi = df['close'] * 1000
    
    def calc_tema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3
    
    mma_tdfi = calc_tema(price_tdfi, mma_length_tdfi)
    smma_tdfi = calc_tema(mma_tdfi, smma_length_tdfi)
    
    impetmma_tdfi = mma_tdfi - mma_tdfi.shift(1)
    impetsmma_tdfi = smma_tdfi - smma_tdfi.shift(1)
    divma_tdfi = np.abs(mma_tdfi - smma_tdfi)
    averimpet_tdfi = (impetmma_tdfi + impetsmma_tdfi) / 2.0
    tdf_tdfi = divma_tdfi * np.power(averimpet_tdfi, n_length_tdfi)
    denom = tdf_tdfi.abs().rolling(lookback_tdfi * n_length_tdfi).max()
    signal_tdfi = tdf_tdfi / denom
    
    signal_long_tdfi = signal_tdfi > filter_high_tdfi
    signal_short_tdfi = signal_tdfi < filter_low_tdfi
    tdfi_long = signal_long_tdfi
    tdfi_short = signal_short_tdfi
    
    # === 3. RANGE FILTER [DW] ===
    rng_qty = 2.618
    rng_per = 14
    smooth_range = True
    smooth_per = 27
    
    h_val = df['close']
    l_val = df['close']
    
    def wilder_ema(series, n):
        alpha = 2.0 / (n + 1)
        result = series.copy()
        result.iloc[n-1] = series.iloc[:n].mean()
        for i in range(n, len(series)):
            result.iloc[i] = series.iloc[i] * alpha + result.iloc[i-1] * (1 - alpha)
        return result
    
    ac_vals = np.abs(h_val - h_val.shift(1))
    ac_ema = wilder_ema(ac_vals, rng_per)
    rng_size = rng_qty * ac_ema
    
    filt = pd.Series(0.0, index=df.index)
    r_ema = wilder_ema(rng_size, smooth_per) if smooth_range else rng_size
    filt.iloc[0] = (h_val.iloc[0] + l_val.iloc[0]) / 2.0
    
    for i in range(1, len(df)):
        r = r_ema.iloc[i]
        h = h_val.iloc[i]
        l = l_val.iloc[i]
        prev_filt = filt.iloc[i-1]
        
        if h - r > prev_filt:
            filt.iloc[i] = h - r
        elif l + r < prev_filt:
            filt.iloc[i] = l + r
        else:
            filt.iloc[i] = prev_filt
    
    fdir = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        if filt.iloc[i] > filt.iloc[i-1]:
            fdir.iloc[i] = 1.0
        elif filt.iloc[i] < filt.iloc[i-1]:
            fdir.iloc[i] = -1.0
        else:
            fdir.iloc[i] = fdir.iloc[i-1]
    
    rf_long = fdir == 1.0
    rf_short = fdir == -1.0
    
    # === 4. ADX (DMI) ===
    adx_length = 14
    adx_threshold = 20.0
    
    plus_dm = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                       np.maximum(df['high'] - df['high'].shift(1), 0), 0)
    minus_dm = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                        np.maximum(df['low'].shift(1) - df['low'], 0), 0)
    
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift(1))
    tr3 = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.Series(np.maximum(np.maximum(tr1, tr2), tr3))
    
    smoothed_plus_dm = pd.Series(plus_dm).copy()
    smoothed_minus_dm = pd.Series(minus_dm).copy()
    smoothed_tr = tr.copy()
    
    alpha = 1.0 / adx_length
    smoothed_plus_dm.iloc[adx_length-1] = plus_dm[:adx_length].mean()
    smoothed_minus_dm.iloc[adx_length-1] = minus_dm[:adx_length].mean()
    smoothed_tr.iloc[adx_length-1] = tr[:adx_length].mean()
    
    for i in range(adx_length, len(df)):
        smoothed_plus_dm.iloc[i] = smoothed_plus_dm.iloc[i-1] - alpha * smoothed_plus_dm.iloc[i-1] + plus_dm[i]
        smoothed_minus_dm.iloc[i] = smoothed_minus_dm.iloc[i-1] - alpha * smoothed_minus_dm.iloc[i-1] + minus_dm[i]
        smoothed_tr.iloc[i] = smoothed_tr.iloc[i-1] - alpha * smoothed_tr.iloc[i-1] + tr.iloc[i]
    
    di_plus = 100.0 * smoothed_plus_dm / smoothed_tr
    di_minus = 100.0 * smoothed_minus_dm / smoothed_tr
    
    dx = 100.0 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    
    adx_val = dx.copy()
    adx_val.iloc[adx_length*2-1] = dx.iloc[adx_length:adx_length*2].mean()
    for i in range(adx_length*2, len(df)):
        adx_val.iloc[i] = adx_val.iloc[i-1] - alpha * adx_val.iloc[i-1] + dx.iloc[i]
    
    adx_ok = adx_val > adx_threshold
    
    # === 5. ENTRY CONDITIONS ===
    long_cond = ctr_long & tdfi_long & rf_long & adx_ok
    short_cond = ctr_short & tdfi_short & rf_short & adx_ok
    
    # === 6. GENERATE ENTRIES ===
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i == 0:
            continue
        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries