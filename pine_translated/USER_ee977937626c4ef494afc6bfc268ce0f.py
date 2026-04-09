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
    
    # Input parameters (matching Pine Script defaults)
    rf_type = 'Type 1'
    rf_mov_src = 'Wicks'
    rf_qty = 2.618
    rf_scale = 'Average Change'
    rf_per = 14
    rf_smooth_range = True
    rf_smooth_per = 27
    rf_av_vals = False
    rf_av_samples = 2
    
    hull_len = 55
    
    h_lookback = 1024
    h_len4 = 32
    h_len5 = 64
    h_len6 = 128
    h_len7 = 256
    h_len8 = 512
    h_smooth_len = 10
    h_threshold = 0.5
    
    atr_len = 14
    
    # Helper functions
    def cond_ema(series, cond, n):
        result = np.full(len(series), np.nan)
        ema_val = np.nan
        for i in range(len(series)):
            if cond.iloc[i] if hasattr(cond, 'iloc') else cond[i]:
                if np.isnan(ema_val):
                    ema_val = series.iloc[i] if hasattr(series, 'iloc') else series[i]
                else:
                    val = series.iloc[i] if hasattr(series, 'iloc') else series[i]
                    ema_val = (val - ema_val) * (2.0 / (n + 1.0)) + ema_val
                result[i] = ema_val
        return pd.Series(result)
    
    def cond_sma(series, cond, n):
        result = np.full(len(series), np.nan)
        vals = []
        for i in range(len(series)):
            if cond.iloc[i] if hasattr(cond, 'iloc') else cond[i]:
                vals.append(series.iloc[i] if hasattr(series, 'iloc') else series[i])
                if len(vals) > n:
                    vals.pop(0)
                result[i] = np.mean(vals) if len(vals) > 0 else np.nan
        return pd.Series(result)
    
    def stdev_func(x, n):
        return np.sqrt(cond_sma(x**2, pd.Series([True]*len(x)), n) - cond_sma(x, pd.Series([True]*len(x)), n)**2)
    
    def rng_size(x, scale, qty, n):
        atr_ = cond_ema(pd.Series(np.abs(df['high'].values - df['low'].values)), pd.Series([True]*len(df)), n)
        ac = cond_ema(pd.Series(np.abs(x.values - x.shift(1).values)), pd.Series([True]*len(x)), n)
        sd = stdev_func(x, n)
        
        result = np.full(len(x), np.nan)
        for i in range(len(x)):
            if scale == 'Pips':
                result[i] = qty * 0.0001
            elif scale == 'Points':
                result[i] = qty * 1.0
            elif scale == '% of Price':
                result[i] = x.iloc[i] * qty / 100.0
            elif scale == 'ATR':
                result[i] = qty * atr_.iloc[i]
            elif scale == 'Average Change':
                result[i] = qty * ac.iloc[i]
            elif scale == 'Standard Deviation':
                result[i] = qty * sd.iloc[i]
            elif scale == 'Ticks':
                result[i] = qty * 0.01
            else:
                result[i] = qty
        return pd.Series(result)
    
    def rng_filt_calc(h_val, l_val, rng_, n, filt_type, smooth, sn, av_rf, av_n):
        rng_smooth = cond_ema(rng_, pd.Series([True]*len(rng_)), sn)
        r = rng_smooth if smooth else rng_
        
        rfilt = np.full(len(r), np.nan)
        rfilt[0] = (h_val.iloc[0] + l_val.iloc[0]) / 2.0
        
        for i in range(1, len(r)):
            prev_rfilt = rfilt[i-1]
            curr_r = r.iloc[i]
            curr_h = h_val.iloc[i]
            curr_l = l_val.iloc[i]
            
            if filt_type == 'Type 1':
                if curr_h - curr_r > prev_rfilt:
                    rfilt[i] = curr_h - curr_r
                elif curr_l + curr_r < prev_rfilt:
                    rfilt[i] = curr_l + curr_r
                else:
                    rfilt[i] = prev_rfilt
            else:
                if curr_h >= prev_rfilt + curr_r:
                    rfilt[i] = prev_rfilt + np.floor(np.abs(curr_h - prev_rfilt) / curr_r) * curr_r
                elif curr_l <= prev_rfilt - curr_r:
                    rfilt[i] = prev_rfilt - np.floor(np.abs(curr_l - prev_rfilt) / curr_r) * curr_r
                else:
                    rfilt[i] = prev_rfilt
        
        rng_filt1 = pd.Series(rfilt)
        hi_band1 = rng_filt1 + r
        lo_band1 = rng_filt1 - r
        
        rng_filt2 = cond_ema(rng_filt1, rng_filt1 != rng_filt1.shift(1), av_n)
        hi_band2 = cond_ema(hi_band1, rng_filt1 != rng_filt1.shift(1), av_n)
        lo_band2 = cond_ema(lo_band1, rng_filt1 != rng_filt1.shift(1), av_n)
        
        rng_filt_ = rng_filt2 if av_rf else rng_filt1
        hi_band_ = hi_band2 if av_rf else hi_band1
        lo_band_ = lo_band2 if av_rf else lo_band1
        
        return hi_band_, lo_band_, rng_filt_
    
    def hull_ma(src, length):
        length_f = float(length)
        half_len = max(1, int(round(length_f / 2.0)))
        sqrt_len = max(1, int(round(np.sqrt(length_f))))
        
        wma2 = src.rolling(half_len).apply(lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
        wma1 = src.rolling(length).apply(lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
        raw = 2.0 * wma2 - wma1
        hull = raw.rolling(sqrt_len).apply(lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True)
        return hull
    
    def get_avg_rs(group_len, pnl_series, lookback):
        results = []
        groups = int(np.floor(lookback / group_len))
        pnl_array = pnl_series.values
        
        for group in range(groups):
            group_shift = group * group_len
            pnl_sum = 0.0
            for i in range(group_len):
                if group_shift + i < len(pnl_array):
                    pnl_sum += pnl_array[group_shift + i]
            
            arr_mean = pnl_sum / group_len if group_len > 0 else 0
            
            dev_sum = 0.0
            for i in range(group_len):
                if group_shift + i < len(pnl_array):
                    dev = pnl_array[group_shift + i] - arr_mean
                    dev_sum += dev * dev
            sd = np.sqrt(dev_sum / (group_len - 1)) if group_len > 1 else 1
            
            cum = 0.0
            cum_min = 1e9
            cum_max = -1e9
            for i in range(group_len):
                if group_shift + i < len(pnl_array):
                    cum += pnl_array[group_shift + i] - arr_mean
                    cum_min = min(cum_min, cum)
                    cum_max = max(cum_max, cum)
            
            if sd > 0:
                results.append((cum_max - cum_min) / sd)
        
        return np.mean(results) if len(results) > 0 else 0
    
    def hurst_exp(pnl_series, lookback):
        log_rs = []
        log_n = []
        
        if h_len4 > 0:
            avg_rs = get_avg_rs(h_len4, pnl_series, lookback)
            if avg_rs > 0:
                log_rs.append(np.log(avg_rs))
                log_n.append(np.log(h_len4))
        
        if h_len5 > 0:
            avg_rs = get_avg_rs(h_len5, pnl_series, lookback)
            if avg_rs > 0:
                log_rs.append(np.log(avg_rs))
                log_n.append(np.log(h_len5))
        
        if h_len6 > 0:
            avg_rs = get_avg_rs(h_len6, pnl_series, lookback)
            if avg_rs > 0:
                log_rs.append(np.log(avg_rs))
                log_n.append(np.log(h_len6))
        
        if h_len7 > 0:
            avg_rs = get_avg_rs(h_len7, pnl_series, lookback)
            if avg_rs > 0:
                log_rs.append(np.log(avg_rs))
                log_n.append(np.log(h_len7))
        
        if h_len8 > 0:
            avg_rs = get_avg_rs(h_len8, pnl_series, lookback)
            if avg_rs > 0:
                log_rs.append(np.log(avg_rs))
                log_n.append(np.log(h_len8))
        
        if len(log_rs) < 2:
            return pd.Series([np.nan] * len(pnl_series))
        
        log_rs = np.array(log_rs)
        log_n = np.array(log_n)
        
        sum_top = 0.0
        sum_bot = 0.0
        mean_log_rs = np.mean(log_rs)
        mean_log_n = np.mean(log_n)
        
        for i in range(len(log_rs)):
            sum_top += (log_rs[i] - mean_log_rs) * (log_n[i] - mean_log_n)
            sum_bot += (log_n[i] - mean_log_n) * (log_n[i] - mean_log_n)
        
        hurst = sum_top / sum_bot if sum_bot != 0 else 0.5
        return pd.Series([hurst] * len(pnl_series))
    
    # Calculations
    h_val = df['high'] if rf_mov_src == 'Wicks' else df['close']
    l_val = df['low'] if rf_mov_src == 'Wicks' else df['close']
    
    mid_val = (h_val + l_val) / 2.0
    rng_vals = rng_size(mid_val, rf_scale, rf_qty, rf_per)
    
    hi_band, lo_band, filt = rng_filt_calc(h_val, l_val, rng_vals, rf_per, rf_type, rf_smooth_range, rf_smooth_per, rf_av_vals, rf_av_samples)
    
    # fdir calculation
    fdir = np.zeros(len(filt))
    fdir[0] = 0.0
    for i in range(1, len(filt)):
        if filt.iloc[i] > filt.iloc[i-1]:
            fdir[i] = 1.0
        elif filt.iloc[i] < filt.iloc[i-1]:
            fdir[i] = -1.0
        else:
            fdir[i] = fdir[i-1]
    fdir = pd.Series(fdir)
    
    rf_bull = (fdir == 1.0) & (df['close'] > filt)
    rf_bear = (fdir == -1.0) & (df['close'] < filt)
    
    # Hull MA
    hlc3 = (df['high'] + df['low'] + df['close']) / 3.0
    hull = hull_ma(hlc3, hull_len)
    
    hull_bull = df['close'] > hull
    hull_bear = df['close'] < hull
    
    hull_buy = (df['close'] > hull) & (df['close'].shift(1) <= hull.shift(1))
    hull_sell = (df['close'] < hull) & (df['close'].shift(1) >= hull.shift(1))
    
    # Hurst regime
    pnl = df['close'] / df['close'].shift(1) - 1.0
    hurst_raw = hurst_exp(pnl, h_lookback)
    hurst_smooth = hurst_raw.ewm(span=h_smooth_len, adjust=False).mean()
    trending_regime = hurst_smooth > h_threshold
    
    # Entry conditions
    long_condition = rf_bull & hull_bull & hull_buy & trending_regime
    short_condition = rf_bear & hull_bear & hull_sell & trending_regime
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if long_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000.0, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000.0, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries