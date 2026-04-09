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
    
    # Settings (matching Pine Script inputs)
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
    
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Wilder ATR
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_val = tr.rolling(window=lookback_bars).mean()
    for i in range(lookback_bars, len(tr)):
        if pd.notna(atr_val.iloc[i-1]):
            atr_val.iloc[i] = (atr_val.iloc[i-1] * (lookback_bars - 1) + tr.iloc[i]) / lookback_bars
    
    # Elephant Candle
    is_green = close > open_
    is_red = close < open_
    body_size = (close - open_).abs()
    total_range = (high - low).abs()
    body_pct = total_range.replace(0, np.nan).fillna(0)
    body_pct = body_size * 100 / total_range.replace(0, np.nan)
    body_pct = body_pct.fillna(0)
    atr_prev = atr_val.shift(1)
    elephant_green = is_green & (body_pct >= min_body_pct) & (body_size >= atr_prev * atr_factor)
    elephant_red = is_red & (body_pct >= min_body_pct) & (body_size >= atr_prev * atr_factor)
    
    # Donchian Channel
    dc_upper = high.rolling(window=dc_length).max()
    dc_lower = low.rolling(window=dc_length).min()
    breakout_long = close > dc_upper.shift(1)
    breakout_short = close < dc_lower.shift(1)
    
    # Hurst Exponent
    pnl = close / close.shift(1) - 1
    log_rs_list = []
    log_n_list = []
    
    def get_avg_rs(group_len):
        if group_len <= 0:
            return np.nan
        groups = int(hurst_lookback // group_len)
        res_ran = []
        for group in range(groups):
            group_shift = group * group_len
            pnl_vals = pnl.iloc[group_shift:group_shift + group_len].values
            if len(pnl_vals) < group_len:
                continue
            pnl_sum = np.nansum(pnl_vals)
            arr_mean = pnl_sum / group_len if not np.isnan(pnl_sum) else np.nan
            if np.isnan(arr_mean):
                continue
            dev_sum = np.nansum((pnl_vals - arr_mean) ** 2)
            sd = np.sqrt(dev_sum / (group_len - 1)) if group_len > 1 else 0
            if sd > 0:
                cum = 0.0
                cum_min = 999999999.0
                cum_max = -999999999.0
                for i in range(group_len):
                    cum += pnl_vals[i] - arr_mean
                    cum_min = min(cum_min, cum)
                    cum_max = max(cum_max, cum)
                res_ran.append((cum_max - cum_min) / sd)
        return np.mean(res_ran) if res_ran else np.nan
    
    avg_rs4 = get_avg_rs(hurt_len4)
    if not np.isnan(avg_rs4) and avg_rs4 > 0:
        log_rs_list.append(np.log(avg_rs4))
        log_n_list.append(np.log(hurt_len4))
    avg_rs5 = get_avg_rs(hurt_len5)
    if not np.isnan(avg_rs5) and avg_rs5 > 0:
        log_rs_list.append(np.log(avg_rs5))
        log_n_list.append(np.log(hurt_len5))
    avg_rs6 = get_avg_rs(hurt_len6)
    if not np.isnan(avg_rs6) and avg_rs6 > 0:
        log_rs_list.append(np.log(avg_rs6))
        log_n_list.append(np.log(hurt_len6))
    avg_rs7 = get_avg_rs(hurt_len7)
    if not np.isnan(avg_rs7) and avg_rs7 > 0:
        log_rs_list.append(np.log(avg_rs7))
        log_n_list.append(np.log(hurt_len7))
    avg_rs8 = get_avg_rs(hurt_len8)
    if not np.isnan(avg_rs8) and avg_rs8 > 0:
        log_rs_list.append(np.log(avg_rs8))
        log_n_list.append(np.log(hurt_len8))
    
    hurstexp = pd.Series([np.nan] * len(df))
    if len(log_rs_list) > 0:
        log_rs_arr = np.array(log_rs_list)
        log_n_arr = np.array(log_n_list)
        avg_log_rs = np.mean(log_rs_arr)
        avg_log_n = np.mean(log_n_arr)
        sum_top = np.sum((log_rs_arr - avg_log_rs) * (log_n_arr - avg_log_n))
        sum_bot = np.sum((log_n_arr - avg_log_n) ** 2)
        hurst_val = sum_top / sum_bot if sum_bot != 0 else np.nan
        hurstexp = pd.Series([hurst_val] * len(df))
    
    hurstexp_smooth = hurstexp.ewm(span=smooth_len, adjust=False).mean() if smooth_hurst else hurstexp
    hurst_consolidated = (hurstexp_smooth < hurst_threshold) if smooth_hurst else (hurstexp < hurst_threshold)
    
    # VFI
    typical = (high + low + close) / 3
    inter = np.log(typical) - np.log(typical.shift(1))
    vinter = inter.rolling(window=30).std()
    cutoff = vfi_coef * vinter * close
    vave = volume.rolling(window=vfi_length).mean().shift(1)
    vmax = vave * vfi_vcoef
    vc = volume.where(volume < vmax, vmax)
    mf = typical - typical.shift(1)
    vcp = np.where(mf > cutoff, vc, np.where(mf < -cutoff, -vc, 0))
    vcp_series = pd.Series(vcp, index=df.index)
    vfi_nom = vcp_series.rolling(window=vfi_length).sum() / vave
    vfi = vfi_nom.rolling(window=3).mean()
    vfima = vfi.ewm(span=vfi_signal, adjust=False).mean()
    vfi_bullish = (vfi > vfima) & (vfi > 0)
    
    # Entry Conditions
    long_condition = breakout_long & \
                     ((~enable_elephant) | elephant_green) & \
                     ((~enable_hurst) | hurst_consolidated) & \
                     ((~enable_vfi) | vfi_bullish)
    
    short_condition = breakout_short & \
                      ((~enable_elephant) | elephant_red) & \
                      ((~enable_hurst) | hurst_consolidated) & \
                      ((~enable_vfi) | (~vfi_bullish))
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(atr_val.iloc[i]) or pd.isna(dc_upper.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
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
        elif short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
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
    
    return entries