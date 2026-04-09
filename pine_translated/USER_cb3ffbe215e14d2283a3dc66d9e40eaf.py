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
    open_col = df['open']
    
    baseline_length = 14
    baseline_source = close
    white_line_length = 20
    ttms_length = 20
    ttms_bb_mult = 2.0
    ttms_kc_mult_1 = 1.0
    ttms_kc_mult_2 = 1.5
    ttms_kc_mult_3 = 2.0
    tether_fast_length = 13
    tether_slow_length = 55
    vortex_length = 14
    vortex_threshold = 0.05
    use_ttms = True
    ttms_green_red = True
    ttms_crossing = True
    ttms_inverse = False
    ttms_highlight = True
    vortex_use = True
    
    mg = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        src_val = baseline_source.iloc[i]
        prev_mg = mg.iloc[i-1] if i > 0 else np.nan
        if pd.isna(prev_mg):
            mg.iloc[i] = src_val.ewm(span=baseline_length, adjust=False).mean().iloc[i] if i > 0 else src_val
        else:
            mg.iloc[i] = prev_mg + (src_val - prev_mg) / (baseline_length * pow(src_val / prev_mg, 4))
    baseline_val = mg
    
    highest = high.rolling(white_line_length).max()
    lowest = low.rolling(white_line_length).min()
    white_line_val = (highest + lowest) / 2
    
    ttms_basis = close.rolling(ttms_length).mean()
    ttms_dev = close.rolling(ttms_length).std()
    bb_upper = ttms_basis + ttms_bb_mult * ttms_dev
    bb_lower = ttms_basis - ttms_bb_mult * ttms_dev
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    kc_dev = tr.rolling(ttms_length).mean()
    kc_upper_3 = ttms_basis + kc_dev * ttms_kc_mult_3
    kc_lower_3 = ttms_basis - kc_dev * ttms_kc_mult_3
    kc_upper_2 = ttms_basis + kc_dev * ttms_kc_mult_2
    kc_lower_2 = ttms_basis - kc_dev * ttms_kc_mult_2
    kc_upper_1 = ttms_basis + kc_dev * ttms_kc_mult_1
    kc_lower_1 = ttms_basis - kc_dev * ttms_kc_mult_1
    ttms_no_squeeze = (bb_lower < kc_lower_3) | (bb_upper > kc_upper_3)
    ttms_high_squeeze = (bb_lower >= kc_lower_1) | (bb_upper <= kc_upper_1)
    highest_high = high.rolling(ttms_length).max()
    lowest_low = low.rolling(ttms_length).min()
    price_avg = (highest_high + lowest_low + ttms_basis) / 3
    momentum = pd.Series(index=df.index, dtype=float)
    for i in range(ttms_length - 1, len(df)):
        y = (close.iloc[i-ttms_length+1:i+1] - price_avg.iloc[i-ttms_length+1:i+1]).values
        x = np.arange(ttms_length)
        if len(y) == ttms_length and not np.any(np.isnan(y)):
            coeffs = np.polyfit(x, y, 1)
            momentum.iloc[i] = coeffs[0]
    momentum_diff = momentum.diff()
    ttms_signals = pd.Series(index=df.index, dtype=float)
    for i in range(1, len(df)):
        if pd.isna(momentum.iloc[i]) or pd.isna(momentum.iloc[i-1]):
            ttms_signals.iloc[i] = 0
        elif momentum.iloc[i] > 0 and momentum.iloc[i] > momentum.iloc[i-1]:
            ttms_signals.iloc[i] = 1
        elif momentum.iloc[i] > 0 and momentum.iloc[i] < momentum.iloc[i-1]:
            ttms_signals.iloc[i] = 2
        elif momentum.iloc[i] < 0 and momentum.iloc[i] < momentum.iloc[i-1]:
            ttms_signals.iloc[i] = -1
        elif momentum.iloc[i] < 0 and momentum.iloc[i] > momentum.iloc[i-1]:
            ttms_signals.iloc[i] = -2
        else:
            ttms_signals.iloc[i] = 0
    ttms_basic_long = ttms_signals == 1 if ttms_green_red else ttms_signals > 0
    ttms_basic_short = ttms_signals == -1 if ttms_green_red else ttms_signals < 0
    ttms_long_signal = ttms_no_squeeze & ttms_basic_long if ttms_highlight else ttms_basic_long
    ttms_short_signal = ttms_no_squeeze & ttms_basic_short if ttms_highlight else ttms_basic_short
    ttms_long_cross = (~ttms_long_signal.shift(1).fillna(False)) & ttms_long_signal if ttms_crossing else ttms_long_signal
    ttms_short_cross = (~ttms_short_signal.shift(1).fillna(False)) & ttms_short_signal if ttms_crossing else ttms_short_signal
    ttms_long_final = ttms_long_cross if use_ttms and not ttms_inverse else (ttms_short_cross if use_ttms and ttms_inverse else True)
    ttms_short_final = ttms_short_cross if use_ttms and not ttms_inverse else (ttms_long_cross if use_ttms and ttms_inverse else True)
    
    tether_high_fast = high.rolling(tether_fast_length).max()
    tether_low_fast = low.rolling(tether_fast_length).min()
    tether_fast_line = (tether_high_fast + tether_low_fast) / 2
    tether_high_slow = high.rolling(tether_slow_length).max()
    tether_low_slow = low.rolling(tether_slow_length).min()
    tether_slow_line = (tether_high_slow + tether_low_slow) / 2
    tether_long = (tether_fast_line > tether_slow_line) & (tether_fast_line.shift(1) <= tether_slow_line.shift(1))
    tether_short = (tether_fast_line < tether_slow_line) & (tether_fast_line.shift(1) >= tether_slow_line.shift(1))
    
    vortex_plus_arr = np.zeros(len(df))
    vortex_minus_arr = np.zeros(len(df))
    for i in range(1, len(df)):
        if i >= vortex_length:
            vp = np.sum(np.abs(high.iloc[i-vortex_length+1:i+1].values - low.shift(1).iloc[i-vortex_length+1:i+1].values))
            vm = np.sum(np.abs(low.iloc[i-vortex_length+1:i+1].values - high.shift(1).iloc[i-vortex_length+1:i+1].values))
            vortex_plus_arr[i] = vp
            vortex_minus_arr[i] = vm
    vortex_plus = pd.Series(vortex_plus_arr, index=df.index)
    vortex_minus = pd.Series(vortex_minus_arr, index=df.index)
    vortex_sum = tr.rolling(vortex_length).sum()
    vortex_pos = vortex_plus / vortex_sum
    vortex_neg = vortex_minus / vortex_sum
    vortex_diff = np.abs(vortex_pos - vortex_neg)
    vortex_confirm = vortex_diff > vortex_threshold if vortex_use else pd.Series(True, index=df.index)
    
    baseline_trend = close > baseline_val
    white_line_trend = close > white_line_val
    bullish_long = vortex_confirm & baseline_trend & white_line_trend & ttms_long_final & tether_long
    bearish_short = vortex_confirm & (~baseline_trend) & (~white_line_trend) & ttms_short_final & tether_short
    
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if bullish_long.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif bearish_short.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    return entries