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
    
    # Input parameters
    baseline_adx_length = 2
    baseline_weight = 10.0
    baseline_ma_length = 6
    
    conf1_smooth = 21
    conf1_constant = 0.4
    conf1_cross = True
    conf1_inverse = False
    
    conf2_poles = 4
    conf2_period = 144
    conf2_multiplier = 1.414
    conf2_cross = True
    conf2_inverse = False
    
    vol_ma_length = 100
    vol_length = 60
    vol_smooth = 3
    vol_threshold = 90
    
    # Compute indicators
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Quantum Adaptive MA
    power_bulls_list = []
    power_bears_list = []
    str_range_list = []
    adx_val_list = []
    
    prev_power_bulls = 0.0
    prev_power_bears = 0.0
    prev_str_range = 0.0
    prev_var_ma = close.iloc[0] if len(close) > 0 else 0.0
    
    for i in range(len(df)):
        hi = high.iloc[i]
        lo = low.iloc[i]
        close_curr = close.iloc[i]
        
        hi1 = high.iloc[i-1] if i > 0 else hi
        lo1 = low.iloc[i-1] if i > 0 else lo
        close1 = close.iloc[i-1] if i > 0 else close_curr
        
        bulls1 = 0.5 * (abs(hi - hi1) + (hi - hi1))
        bears1 = 0.5 * (abs(lo1 - lo) + (lo1 - lo))
        
        if bulls1 > bears1:
            bulls = 0
            bears = bears1
        elif bulls1 < bears1:
            bulls = bulls1
            bears = 0
        else:
            bulls = 0
            bears = 0
        
        if i > 0:
            power_bulls = (baseline_weight * prev_power_bulls + bulls) / (baseline_weight + 1)
            power_bears = (baseline_weight * prev_power_bears + bears) / (baseline_weight + 1)
        else:
            power_bulls = 0.0
            power_bears = 0.0
        
        prev_power_bulls = power_bulls
        prev_power_bears = power_bears
        power_bulls_list.append(power_bulls)
        power_bears_list.append(power_bears)
        
        true_range = max(hi - lo, hi - close1)
        
        if i > 0:
            str_range = (baseline_weight * prev_str_range + true_range) / (baseline_weight + 1)
        else:
            str_range = true_range
        
        prev_str_range = str_range
        str_range_list.append(str_range)
        
        pos_di = power_bulls / str_range if str_range > 0 else 0
        neg_di = power_bears / str_range if str_range > 0 else 0
        di_diff = abs(pos_di - neg_di) / (pos_di + neg_di) if (pos_di + neg_di) > 0 else 0
        
        if i > 0:
            diag_x = (baseline_weight * adx_val_list[i-1] + di_diff) / (baseline_weight + 1)
        else:
            diag_x = di_diff
        
        adx_val = diag_x
        adx_val_list.append(adx_val)
    
    adx_series = pd.Series(adx_val_list, index=df.index)
    
    adx_low_vals = adx_series.rolling(baseline_adx_length).min()
    adx_high_vals = adx_series.rolling(baseline_adx_length).max()
    adx_min_vals = adx_low_vals.clip(lower=None, upper=1000000.0)
    adx_max_vals = adx_high_vals.clip(lower=-1.0, upper=None)
    adx_diff_vals = adx_max_vals - adx_min_vals
    adx_constant_series = np.where(adx_diff_vals > 0, (adx_series - adx_min_vals) / adx_diff_vals, 0)
    
    var_ma_list = []
    prev_var_ma = close.iloc[0] if len(close) > 0 else 0.0
    
    for i in range(len(df)):
        if i > 0:
            var_ma = ((2 - adx_constant_series.iloc[i]) * prev_var_ma + adx_constant_series.iloc[i] * close.iloc[i]) / 2
        else:
            var_ma = close.iloc[i]
        prev_var_ma = var_ma
        var_ma_list.append(var_ma)
    
    var_ma_series = pd.Series(var_ma_list, index=df.index)
    baseline_ma = var_ma_series.rolling(baseline_ma_length).mean()
    
    # WavePulse (CoralChannel Based)
    di_val = (conf1_smooth - 1.0) / 2.0 + 1.0
    c1_val = 2 / (di_val + 1.0)
    c2_val = 1 - c1_val
    c3_val = 3.0 * (conf1_constant ** 2 + conf1_constant ** 3)
    c4_val = -3.0 * (2.0 * conf1_constant ** 2 + conf1_constant + conf1_constant ** 3)
    c5_val = 3.0 * conf1_constant + 1.0 + conf1_constant ** 3 + 3.0 * conf1_constant ** 2
    
    wave_val_list = []
    i1_vals = []
    i2_vals = []
    i3_vals = []
    i4_vals = []
    i5_vals = []
    i6_vals = []
    
    prev_i1 = 0.0
    prev_i2 = 0.0
    prev_i3 = 0.0
    prev_i4 = 0.0
    prev_i5 = 0.0
    prev_i6 = 0.0
    
    for i in range(len(df)):
        i1_val = c1_val * close.iloc[i] + c2_val * prev_i1
        i2_val = c1_val * i1_val + c2_val * prev_i2
        i3_val = c1_val * i2_val + c2_val * prev_i3
        i4_val = c1_val * i3_val + c2_val * prev_i4
        i5_val = c1_val * i4_val + c2_val * prev_i5
        i6_val = c1_val * i5_val + c2_val * prev_i6
        
        wave_val = -conf1_constant ** 3 * i6_val + c3_val * i5_val + c4_val * i4_val + c5_val * i3_val
        
        prev_i1 = i1_val
        prev_i2 = i2_val
        prev_i3 = i3_val
        prev_i4 = i4_val
        prev_i5 = i5_val
        prev_i6 = i6_val
        
        i1_vals.append(i1_val)
        i2_vals.append(i2_val)
        i3_vals.append(i3_val)
        i4_vals.append(i4_val)
        i5_vals.append(i5_val)
        i6_vals.append(i6_val)
        wave_val_list.append(wave_val)
    
    wave_val = pd.Series(wave_val_list, index=df.index)
    
    # Quantum Channel Filter
    def f_quantum_filter(source, alpha, pole_idx):
        x = 1 - alpha
        m2 = 36 if pole_idx == 9 else 28 if pole_idx == 8 else 21 if pole_idx == 7 else 15 if pole_idx == 6 else 10 if pole_idx == 5 else 6 if pole_idx == 4 else 3 if pole_idx == 3 else 1 if pole_idx == 2 else 0
        m3 = 84 if pole_idx == 9 else 56 if pole_idx == 8 else 35 if pole_idx == 7 else 20 if pole_idx == 6 else 10 if pole_idx == 5 else 4 if pole_idx == 4 else 1 if pole_idx == 3 else 0
        m4 = 126 if pole_idx == 9 else 70 if pole_idx == 8 else 35 if pole_idx == 7 else 15 if pole_idx == 6 else 5 if pole_idx == 5 else 1 if pole_idx == 4 else 0
        m5 = 126 if pole_idx == 9 else 56 if pole_idx == 8 else 21 if pole_idx == 7 else 6 if pole_idx == 6 else 1 if pole_idx == 5 else 0
        m6 = 84 if pole_idx == 9 else 28 if pole_idx == 8 else 7 if pole_idx == 7 else 1 if pole_idx == 6 else 0
        m7 = 36 if pole_idx == 9 else 8 if pole_idx == 8 else 1 if pole_idx == 7 else 0
        m8 = 9 if pole_idx == 9 else 1 if pole_idx == 8 else 0
        m9 = 1 if pole_idx == 9 else 0
        
        f = alpha ** pole_idx * source
        f += pole_idx * x * 0 if pole_idx < 1 else 0
        if pole_idx >= 2:
            f -= m2 * x ** 2 * 0
        if pole_idx >= 3:
            f += m3 * x ** 3 * 0
        if pole_idx >= 4:
            f -= m4 * x ** 4 * 0
        if pole_idx >= 5:
            f += m5 * x ** 5 * 0
        if pole_idx >= 6:
            f -= m6 * x ** 6 * 0
        if pole_idx >= 7:
            f += m7 * x ** 7 * 0
        if pole_idx >= 8:
            f -= m8 * x ** 8 * 0
        if pole_idx == 9:
            f += m9 * x ** 9 * 0
        
        return f
    
    alpha = 2.0 / (conf2_period + 1)
    q_channel_upper_list = []
    q_channel_lower_list = []
    
    f_history = [[0.0] * 10 for _ in range(10)]
    
    for i in range(len(df)):
        if i >= conf2_poles:
            src = (high.iloc[i] + low.iloc[i] + close.iloc[i]) / 3.0
            
            f_vals = []
            for pole in range(1, 10):
                if pole <= conf2_poles:
                    x = 1 - alpha
                    m2 = 36 if pole == 9 else 28 if pole == 8 else 21 if pole == 7 else 15 if pole == 6 else 10 if pole == 5 else 6 if pole == 4 else 3 if pole == 3 else 1 if pole == 2 else 0
                    m3 = 84 if pole == 9 else 56 if pole == 8 else 35 if pole == 7 else 20 if pole == 6 else 10 if pole == 5 else 4 if pole == 4 else 1 if pole == 3 else 0
                    m4 = 126 if pole == 9 else 70 if pole == 8 else 35 if pole == 7 else 15 if pole == 6 else 5 if pole == 5 else 1 if pole == 4 else 0
                    m5 = 126 if pole == 9 else 56 if pole == 8 else 21 if pole == 7 else 6 if pole == 6 else 1 if pole == 5 else 0
                    m6 = 84 if pole == 9 else 28 if pole == 8 else 7 if pole == 7 else 1 if pole == 6 else 0
                    m7 = 36 if pole == 9 else 8 if pole == 8 else 1 if pole == 7 else 0
                    m8 = 9 if pole == 9 else 1 if pole == 8 else 0
                    m9 = 1 if pole == 9 else 0
                    
                    prev_s = f_history[pole][i-1] if i-1 >= 0 else 0.0
                    prev_f1 = f_history[1][i-1] if i-1 >= 0 else 0.0
                    prev_f2 = f_history[2][i-1] if i-1 >= 0 else 0.0
                    prev_f3 = f_history[3][i-1] if i-1 >= 0 else 0.0
                    prev_f4 = f_history[4][i-1] if i-1 >= 0 else 0.0
                    prev_f5 = f_history[5][i-1] if i-1 >= 0 else 0.0
                    prev_f6 = f_history[6][i-1] if i-1 >= 0 else 0.0
                    prev_f7 = f_history[7][i-1] if i-1 >= 0 else 0.0
                    prev_f8 = f_history[8][i-1] if i-1 >= 0 else 0.0
                    prev_f9 = f_history[9][i-1] if i-1 >= 0 else 0.0
                    
                    f = alpha ** pole * src
                    f += pole * x * prev_s
                    if pole >= 2:
                        f -= m2 * x ** 2 * prev_f2
                    if pole >= 3:
                        f += m3 * x ** 3 * prev_f3
                    if pole >= 4:
                        f -= m4 * x ** 4 * prev_f4
                    if pole >= 5:
                        f += m5 * x ** 5 * prev_f5
                    if pole >= 6:
                        f -= m6 * x ** 6 * prev_f6
                    if pole >= 7:
                        f += m7 * x ** 7 * prev_f7
                    if pole >= 8:
                        f -= m8 * x ** 8 * prev_f8
                    if pole == 9:
                        f += m9 * x ** 9 * prev_f9
                    
                    f_vals.append(f)
                    f_history[pole][i] = f
                else:
                    f_vals.append(0.0)
            
            fn = f_vals[conf2_poles - 1] if conf2_poles >= 1 else 0.0
            f1 = f_vals[0] if len(f_vals) > 0 else 0.0
            
            q_channel_upper_list.append(fn)
            q_channel_lower_list.append(f1)
        else:
            q_channel_upper_list.append(np.nan)
            q_channel_lower_list.append(np.nan)
            for pole in range(1, 10):
                f_history[pole][i] = 0.0
    
    q_channel_upper = pd.Series(q_channel_upper_list, index=df.index)
    q_channel_lower = pd.Series(q_channel_lower_list, index=df.index)
    
    # Momentum Density
    bound = close.rolling(vol_ma_length).mean() - 0.2 * close.rolling(vol_ma_length).std()
    sum_above = (close > bound).rolling(vol_length).sum()
    density_raw = sum_above * 100 / vol_length
    mom_density = density_raw.ewm(span=vol_smooth, adjust=False).mean()
    
    # Entry conditions
    # Long: cross above baseline, wave_pulse confirmation, momentum density filter
    long_cond = (close > baseline_ma) & (close.shift(1) <= baseline_ma.shift(1))
    
    # Short: cross below baseline, wave_pulse confirmation, momentum density filter
    short_cond = (close < baseline_ma) & (close.shift(1) >= baseline_ma.shift(1))
    
    # Build entries
    entries = []
    trade_num = 1
    
    last_long_bar = -100
    last_short_bar = -100
    
    for i in range(len(df)):
        if i <= max(conf2_poles, vol_ma_length, baseline_ma_length):
            continue
        
        if pd.isna(baseline_ma.iloc[i]) or pd.isna(wave_val.iloc[i]):
            continue
        if pd.isna(q_channel_upper.iloc[i]) or pd.isna(mom_density.iloc[i]):
            continue
        
        # Long entry
        if long_cond.iloc[i] and (i - last_long_bar) > 3:
            if mom_density.iloc[i] < vol_threshold:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
                trade_num += 1
                last_long_bar = i
        
        # Short entry
        if short_cond.iloc[i] and (i - last_short_bar) > 3:
            if mom_density.iloc[i] < vol_threshold:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
                trade_num += 1
                last_short_bar = i
    
    return entries