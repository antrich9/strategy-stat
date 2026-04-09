import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    source = close
    
    def calc_alma(src, window, offset_val, sigma):
        m = offset_val * (window - 1)
        s = window / sigma
        w = np.exp(-((np.arange(window) - m) ** 2) / (2 * s * s))
        w = w / w.sum()
        result = np.convolve(src, w, mode='same')
        return pd.Series(result, index=src.index)
    
    def calc_fluxwave(src, smooth_len, cycle_len, alma_off, alma_sig):
        ema = src.ewm(span=smooth_len, adjust=False).mean()
        roc = ((ema / ema.shift(cycle_len)) - 1) * 100
        smoothed = calc_alma(roc, smooth_len, alma_off, alma_sig)
        signal = smoothed.ewm(span=smooth_len, adjust=False).mean()
        return smoothed, signal
    
    def calc_quicksilver(prices, period, dev_mult):
        mid = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = mid + (std * dev_mult)
        lower = mid - (std * dev_mult)
        return upper, mid, lower
    
    def calc_volume_energy(vol, length):
        roll_max = vol.rolling(length).max()
        roll_min = vol.rolling(length).min()
        norm_vol = (vol - roll_min) / (roll_max - roll_min + 1e-10)
        return norm_vol
    
    def kalman_filter(src, meas_noise, proc_noise):
        result = np.zeros(len(src))
        x = src.iloc[0]
        p = 1.0
        result[0] = x
        for i in range(1, len(src)):
            p = p + proc_noise
            k = p / (p + meas_noise)
            x = x + k * (src.iloc[i] - x)
            p = (1 - k) * p
            result[i] = x
        return pd.Series(result, index=src.index)
    
    def wavelet_filter(src, levels, period):
        result = src.copy()
        for _ in range(levels):
            diff = src.diff(period)
            result = result - diff
        return result
    
    filtered = source
    if True:
        filtered = kalman_filter(source, 0.01, 0.001)
    
    alma_offset = 0.85
    alma_sigma = 6.0
    alma_win = 9
    alma_fast = calc_alma(filtered, alma_win, alma_offset, alma_sigma)
    alma_slow = calc_alma(filtered, alma_win * 2, alma_offset, alma_sigma)
    signal_alma = calc_alma(filtered, alma_win, alma_offset, alma_sigma)
    
    fluxwave_val, fluxwave_signal = calc_fluxwave(filtered, 1, 50, 0.85, 6.0)
    
    qs_upper, qs_mid, qs_lower = calc_quicksilver(filtered, 20, 2.0)
    
    vol_energy = calc_volume_energy(volume, 50)
    
    alma_cross_long = (alma_fast > alma_slow) & (alma_fast.shift(1) <= alma_slow.shift(1))
    alma_cross_short = (alma_fast < alma_slow) & (alma_fast.shift(1) >= alma_slow.shift(1))
    alma_cross = alma_cross_long | alma_cross_short
    
    flux_cross_long = (fluxwave_val < 30) & (fluxwave_val.shift(1) >= 30)
    flux_cross_short = (fluxwave_val > 70) & (fluxwave_val.shift(1) <= 70)
    flux_cross = flux_cross_long | flux_cross_short
    
    qs_break_long = filtered > qs_upper
    qs_break_short = filtered < qs_lower
    
    vol_confirm = vol_energy > 0.5
    
    entry_lookback = 5
    qfn_cooldown = 3
    entry_threshold = 0.5
    use_vol_filter = True
    use_alma_signal = True
    use_flux_signal = True
    use_qs_signal = True
    
    long_base = alma_cross_long.copy()
    short_base = alma_cross_short.copy()
    if use_flux_signal:
        long_base = long_base | flux_cross_long
        short_base = short_base | flux_cross_short
    if use_qs_signal:
        long_base = long_base | qs_break_long
        short_base = short_base | qs_break_short
    
    long_cond = long_base
    short_cond = short_base
    if use_vol_filter:
        long_cond = long_cond & vol_confirm
        short_cond = short_cond & vol_confirm
    
    entries = []
    trade_num = 1
    last_trade_bar = -qfn_cooldown
    
    for i in range(entry_lookback, len(df)):
        if i <= alma_slow.index.get_loc(alma_slow.dropna().index[0]) if len(alma_slow.dropna()) > 0 else entry_lookback:
            continue
        
        lookback_start = i - entry_lookback
        long_in_lookback = long_cond.iloc[lookback_start:i+1].sum() > 0
        short_in_lookback = short_cond.iloc[lookback_start:i+1].sum() > 0
        
        is_long = long_in_lookback and (i - last_trade_bar) > qfn_cooldown
        is_short = short_in_lookback and (i - last_trade_bar) > qfn_cooldown
        
        if is_long:
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
            last_trade_bar = i
        elif is_short:
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
            last_trade_bar = i
    
    return entries