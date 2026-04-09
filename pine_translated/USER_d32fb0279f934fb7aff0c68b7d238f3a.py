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
    open_ = df['open'].copy()
    high = df['high'].copy()
    low = df['low'].copy()
    close = df['close'].copy()
    volume = df['volume'].copy()
    
    # Settings
    enable_elephant = True
    min_body_pct = 70
    lookback_bars = 100
    atr_factor = 1.3
    dc_length = 20
    enable_hurst = True
    hurst_lookback = 512
    hurst_threshold = 0.5
    smooth_hurst = True
    smooth_len = 10
    hurt_len4 = 32
    hurt_len5 = 64
    hurt_len6 = 128
    hurt_len7 = 256
    hurt_len8 = 512
    enable_vfi = True
    vfi_length = 130
    vfi_coef = 0.2
    vfi_vcoef = 2.5
    vfi_signal = 5
    
    n = len(df)
    half_len = n // 2
    
    # Helper: Wilder RSI
    def wilder_rsi(src, length):
        delta = src.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Helper: Wilder ATR
    def wilder_atr(high_arr, low_arr, close_arr, length):
        prev_close = close_arr.shift(1)
        tr1 = high_arr - low_arr
        tr2 = (high_arr - prev_close).abs()
        tr3 = (low_arr - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
        return atr
    
    # Calculate ATR
    atr_val = wilder_atr(high, low, close, lookback_bars)
    
    # Elephant Candle conditions
    is_green = close > open_
    is_red = close < open_
    body_size = (close - open_).abs()
    total_range = (high - low).abs()
    body_pct = np.where(total_range > 0, (body_size * 100) / total_range, 0)
    body_pct_series = pd.Series(body_pct, index=df.index)
    atr_prev = atr_val.shift(1)
    
    elephant_green = is_green & (body_pct_series >= min_body_pct) & (body_size >= atr_prev * atr_factor)
    elephant_red = is_red & (body_pct_series >= min_body_pct) & (body_size >= atr_prev * atr_factor)
    
    # Donchian Channel
    dc_upper = high.rolling(window=dc_length).max()
    dc_lower = low.rolling(window=dc_length).min()
    dc_upper_prev = dc_upper.shift(1)
    dc_lower_prev = dc_lower.shift(1)
    
    breakout_long = close > dc_upper_prev
    breakout_short = close < dc_lower_prev
    
    # Hurst Exponent
    pnl = close / close.shift(1) - 1
    
    def get_avg_rs(group_len, pnl_series):
        groups = int(hurst_lookback // group_len)
        res_ran = []
        for group in range(groups):
            group_shift = group * group_len
            pnl_sum = 0.0
            for i in range(group_len):
                idx = group_shift + i
                if idx < len(pnl_series):
                    pnl_sum += pnl_series.iloc[idx]
            if group_len > 0:
                arr_mean = pnl_sum / group_len
            else:
                arr_mean = 0
            dev_sum = 0.0
            for i in range(group_len):
                idx = group_shift + i
                if idx < len(pnl_series):
                    dev_sum += (pnl_series.iloc[idx] - arr_mean) ** 2
            if group_len > 1:
                sd = np.sqrt(dev_sum / (group_len - 1))
            else:
                sd = 0
            cum = 0.0
            cum_min = 999999999.0
            cum_max = -999999999.0
            for i in range(group_len):
                idx = group_shift + i
                if idx < len(pnl_series):
                    cum += pnl_series.iloc[idx] - arr_mean
                    cum_min = min(cum_min, cum)
                    cum_max = max(cum_max, cum)
            if sd > 0:
                res_ran.append((cum_max - cum_min) / sd)
        if len(res_ran) > 0:
            return np.mean(res_ran)
        return np.nan
    
    log_rs = []
    log_n = []
    
    if hurt_len4 > 0 and n >= hurt_len4:
        avg_rs4 = get_avg_rs(hurt_len4, pnl)
        if not np.isnan(avg_rs4) and avg_rs4 > 0:
            log_rs.append(np.log(avg_rs4))
            log_n.append(np.log(hurt_len4))
    
    if hurt_len5 > 0 and n >= hurt_len5:
        avg_rs5 = get_avg_rs(hurt_len5, pnl)
        if not np.isnan(avg_rs5) and avg_rs5 > 0:
            log_rs.append(np.log(avg_rs5))
            log_n.append(np.log(hurt_len5))
    
    if hurt_len6 > 0 and n >= hurt_len6:
        avg_rs6 = get_avg_rs(hurt_len6, pnl)
        if not np.isnan(avg_rs6) and avg_rs6 > 0:
            log_rs.append(np.log(avg_rs6))
            log_n.append(np.log(hurt_len6))
    
    if hurt_len7 > 0 and n >= hurt_len7:
        avg_rs7 = get_avg_rs(hurt_len7, pnl)
        if not np.isnan(avg_rs7) and avg_rs7 > 0:
            log_rs.append(np.log(avg_rs7))
            log_n.append(np.log(hurt_len7))
    
    if hurt_len8 > 0 and n >= hurt_len8:
        avg_rs8 = get_avg_rs(hurt_len8, pnl)
        if not np.isnan(avg_rs8) and avg_rs8 > 0:
            log_rs.append(np.log(avg_rs8))
            log_n.append(np.log(hurt_len8))
    
    hurstexp_series = pd.Series(index=df.index, dtype=float)
    if len(log_rs) > 1:
        log_rs_arr = np.array(log_rs)
        log_n_arr = np.array(log_n)
        avg_log_rs = np.mean(log_rs_arr)
        avg_log_n = np.mean(log_n_arr)
        sum_top = 0.0
        sum_bot = 0.0
        for i in range(len(log_rs)):
            sum_top += (log_rs_arr[i] - avg_log_rs) * (log_n_arr[i] - avg_log_n)
            sum_bot += (log_n_arr[i] - avg_log_n) ** 2
        hurstexp_val = sum_top / sum_bot if sum_bot != 0 else np.nan
    else:
        hurstexp_val = np.nan
    
    hurstexp_series.iloc[:] = hurstexp_val
    if smooth_hurst:
        hurstexp_smooth_series = hurstexp_series.ewm(span=smooth_len, adjust=False).mean()
    else:
        hurstexp_smooth_series = hurstexp_series.copy()
    
    hurst_consolidated = hurstexp_smooth_series < hurst_threshold
    
    # Volume Flow Indicator
    typical = (high + low + close) / 3.0
    inter = np.log(typical) - np.log(typical.shift(1))
    vinter = inter.rolling(window=30).std()
    cutoff = vfi_coef * vinter * close
    vave = volume.rolling(window=vfi_length).mean().shift(1)
    vmax = vave * vfi_vcoef
    vc = np.minimum(volume, vmax)
    mf = typical - typical.shift(1)
    vcp = np.where(mf > cutoff, vc, np.where(mf < -cutoff, -vc, 0))
    vcp_series = pd.Series(vcp, index=df.index)
    vfi = (vcp_series.rolling(window=vfi_length).sum() / vave).rolling(window=3).mean()
    vfima = vfi.ewm(span=vfi_signal, adjust=False).mean()
    vfi_bullish = (vfi > vfima) & (vfi > 0)
    
    # Entry conditions
    long_condition = breakout_long & \
                     ((not enable_elephant) | elephant_green) & \
                     ((not enable_hurst) | hurst_consolidated) & \
                     ((not enable_vfi) | vfi_bullish)
    
    short_condition = breakout_short & \
                      ((not enable_elephant) | elephant_red) & \
                      ((not enable_hurst) | hurst_consolidated) & \
                      ((not enable_vfi) | (~vfi_bullish))
    
    entries = []
    trade_num = 1
    
    for i in range(1, n):
        if pd.isna(atr_val.iloc[i]) or pd.isna(dc_upper.iloc[i]) or pd.isna(dc_upper_prev.iloc[i]):
            continue
        if pd.isna(atr_prev.iloc[i]):
            continue
        if pd.isna(close.iloc[i]) or pd.isna(open_.iloc[i]) or pd.isna(high.iloc[i]) or pd.isna(low.iloc[i]):
            continue
        if pd.isna(elephant_green.iloc[i] if isinstance(elephant_green.iloc[i], float) else False):
            continue
        
        direction = None
        if long_condition.iloc[i]:
            direction = 'long'
        elif short_condition.iloc[i]:
            direction = 'short'
        
        if direction is not None:
            entry_price = close.iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts / 1000.0, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries