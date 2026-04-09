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
    
    open_ = df['open'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    timestamps = df['time'].values
    
    # Parameters
    entry_method = "Confluence"
    use_trend_filter = True
    trend_length = 50
    min_zones = 2
    require_tsi = False
    left = 20
    right = 15
    atr_len = 30
    mult = 0.5
    per = 5.0
    src = "HA"
    detect_patterns = True
    be_ = True
    hs_ = True
    pc_ = True
    dg_ = True
    longf = 25
    shortf = 5
    signalf = 14
    n_piv = 4
    
    # Heikin Ashi
    ha_close = (open_ + high + low + close) / 4
    ha_open = np.zeros(len(close))
    ha_open[0] = (open_[0] + close[0]) / 2
    for i in range(1, len(close)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    
    hi_ha_bod = np.maximum(close, open_)
    lo_ha_bod = np.minimum(close, open_)
    hi_bod = np.maximum(close, open_)
    lo_bod = np.minimum(close, open_)
    
    if src == "HA":
        src_high = hi_ha_bod
        src_low = lo_ha_bod
    elif src == "High/Low":
        src_high = high
        src_low = low
    else:
        src_high = hi_bod
        src_low = lo_bod
    
    # Pivot High
    pivot_high = np.full(len(close), np.nan)
    for i in range(left, len(close) - right):
        is_high = True
        for j in range(i - left, i + right + 1):
            if j != i and src_high[j] >= src_high[i]:
                is_high = False
                break
        if is_high:
            pivot_high[i] = src_high[i]
    
    # Pivot Low
    pivot_low = np.full(len(close), np.nan)
    for i in range(left, len(close) - right):
        is_low = True
        for j in range(i - left, i + right + 1):
            if j != i and src_low[j] <= src_low[i]:
                is_low = False
                break
        if is_low:
            pivot_low[i] = src_low[i]
    
    # ATR
    tr = np.zeros(len(close))
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(alpha=1/atr_len, adjust=False).mean().values
    
    # Band calculation
    perc_max = close * (per / 100)
    band = np.full(len(close), np.nan)
    for i in range(right, len(close)):
        if not np.isnan(pivot_high[i - right]) or not np.isnan(pivot_low[i - right]):
            band[i] = min(atr[i] * mult, perc_max[i - right]) / 2
    
    # Pivot zones arrays
    high_pivots = []
    high_levels_upper = []
    high_levels_lower = []
    high_is_bull = []
    
    low_pivots = []
    low_levels_upper = []
    low_levels_lower = []
    low_is_bull = []
    
    for i in range(len(close)):
        if not np.isnan(pivot_high[i]) and i >= right:
            new_upper = pivot_high[i] + band[i]
            new_lower = pivot_high[i] - band[i]
            high_pivots.insert(0, i)
            high_levels_upper.insert(0, new_upper)
            high_levels_lower.insert(0, new_lower)
            high_is_bull.insert(0, False)
            if len(high_pivots) > n_piv:
                high_pivots.pop()
                high_levels_upper.pop()
                high_levels_lower.pop()
                high_is_bull.pop()
        
        if not np.isnan(pivot_low[i]) and i >= right:
            new_upper = pivot_low[i] + band[i]
            new_lower = pivot_low[i] - band[i]
            low_pivots.insert(0, i)
            low_levels_upper.insert(0, new_upper)
            low_levels_lower.insert(0, new_lower)
            low_is_bull.insert(0, True)
            if len(low_pivots) > n_piv:
                low_pivots.pop()
                low_levels_upper.pop()
                low_levels_lower.pop()
                low_is_bull.pop()
        
        # Update bull/bear status
        cur_close = close[i]
        for j in range(len(high_pivots)):
            if cur_close > high_levels_upper[j] and not high_is_bull[j]:
                high_is_bull[j] = True
            if cur_close < high_levels_lower[j] and high_is_bull[j]:
                high_is_bull[j] = False
        
        for j in range(len(low_pivots)):
            if cur_close > low_levels_upper[j] and not low_is_bull[j]:
                low_is_bull[j] = True
            if cur_close < low_levels_lower[j] and low_is_bull[j]:
                low_is_bull[j] = False
    
    # Detect function
    def detect_zones():
        fh = False
        hb = False
        for j in range(len(high_pivots)):
            if low[i] < high_levels_upper[j] and high[i] > high_levels_lower[j]:
                fh = True
                hb = high_is_bull[j]
                break
        
        fl = False
        lb = False
        for j in range(len(low_pivots)):
            if low[i] < low_levels_upper[j] and high[i] > low_levels_lower[j]:
                fl = True
                lb = low_is_bull[j]
                break
        
        return fh, hb, fl, lb
    
    # Num level function (replicating Pine bug)
    def num_levels():
        above = 0
        total = 0
        for j in range(len(high_is_bull)):
            if high_is_bull[j]:
                above += 1
            total += 1
        for j in range(len(low_is_bull)):
            if low_is_bull[j]:
                above += 1
            total += 1
        return above, total
    
    # TSI
    price_change = np.diff(close, prepend=np.nan)
    double_smooth_pc = pd.Series(price_change).ewm(span=shortf, adjust=False).mean().ewm(span=longf, adjust=False).mean().values
    abs_price_change = np.abs(price_change)
    double_smooth_abs_pc = pd.Series(abs_price_change).ewm(span=shortf, adjust=False).mean().ewm(span=longf, adjust=False).mean().values
    tsi = 100 * (double_smooth_pc / double_smooth_abs_pc)
    tsl_series = pd.Series(tsi).ewm(span=signalf, adjust=False).mean()
    tsl = tsl_series.values
    
    # Trend EMA
    trend_ema = pd.Series(close).ewm(span=trend_length, adjust=False).mean().values
    up_trend = close > trend_ema
    down_trend = close < trend_ema
    
    # Patterns
    body = np.abs(open_ - close)
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    bar_range = high - low
    
    hammer = be_ & (lower_wick >= 2 * body) & (close > open_) & ((close - np.minimum(open_, close)) < body * 0.1) & (bar_range >= 0.1 * atr)
    star = hs_ & (upper_wick >= 1.5 * body) & (close < open_) & ((np.maximum(open_, close) - close) < body * 0.1) & (bar_range >= 0.1 * atr)
    
    prev_open = np.roll(open_, 1)
    prev_close = np.roll(close, 1)
    prev_body = np.abs(prev_open - prev_close)
    prev_upper_wick = np.roll(high, 1) - np.maximum(prev_open, prev_close)
    prev_lower_wick = np.minimum(prev_open, prev_close) - np.roll(low, 1)
    prev_bar_range = np.roll(high, 1) - np.roll(low, 1)
    
    bull_eng = be_ & (close > open_) & (prev_close < prev_open) & (close >= prev_open) & (open_ <= prev_close) & (close - open_ >= prev_body) & (prev_upper_wick <= 0) & (upper_wick <= 0)
    bear_eng = be_ & (close < open_) & (prev_close > prev_open) & (close <= prev_open) & (open_ >= prev_close) & (open_ - close >= prev_body) & (prev_lower_wick <= 0) & (lower_wick <= 0)
    
    piercing = pc_ & (close > open_) & (prev_close < prev_open) & (open_ < prev_close) & (close > prev_open + prev_body * 0.5) & (close < prev_open)
    dark_cloud = pc_ & (close < open_) & (prev_close > prev_open) & (open_ > prev_close) & (close < prev_open - prev_body * 0.5) & (close > prev_open)
    
    dragon = dg_ & (body < 0.05 * bar_range) & (body > 0)
    grave = dg_ & (body < 0.05 * bar_range) & (body > 0)
    
    bull_pattern = hammer | bull_eng | piercing | dragon
    bear_pattern = star | bear_eng | dark_cloud | grave
    
    # Track levels for false breaks
    track_high_arr = np.zeros(len(close))
    track_low_arr = np.zeros(len(close))
    
    prev_track_high = 0
    prev_track_low = 0
    
    # False breaks detection
    lowest_low_3 = pd.Series(low).rolling(3).min().values
    highest_high_3 = pd.Series(high).rolling(3).max().values
    
    entries = []
    trade_num = 1
    
    for i in range(len(close)):
        if i < 1:
            continue
        
        if not np.isnan(pivot_high[i]) and i >= right:
            new_upper = pivot_high[i] + band[i]
            new_lower = pivot_high[i] - band[i]
            high_pivots.insert(0, i)
            high_levels_upper.insert(0, new_upper)
            high_levels_lower.insert(0, new_lower)
            high_is_bull.insert(0, False)
            if len(high_pivots) > n_piv:
                high_pivots.pop()
                high_levels_upper.pop()
                high_levels_lower.pop()
                high_is_bull.pop()
        
        if not np.isnan(pivot_low[i]) and i >= right:
            new_upper = pivot_low[i] + band[i]
            new_lower = pivot_low[i] - band[i]
            low_pivots.insert(0, i)
            low_levels_upper.insert(0, new_upper)
            low_levels_lower.insert(0, new_lower)
            low_is_bull.insert(0, True)
            if len(low_pivots) > n_piv:
                low_pivots.pop()
                low_levels_upper.pop()
                low_levels_lower.pop()
                low_is_bull.pop()
        
        cur_close = close[i]
        for j in range(len(high_pivots)):
            if cur_close > high_levels_upper[j] and not high_is_bull[j]:
                high_is_bull[j] = True
            if cur_close < high_levels_lower[j] and high_is_bull[j]:
                high_is_bull[j] = False
        
        for j in range(len(low_pivots)):
            if cur_close > low_levels_upper[j] and not low_is_bull[j]:
                low_is_bull[j] = True
            if cur_close < low_levels_lower[j] and low_is_bull[j]:
                low_is_bull[j] = False
        
        fh, hb, fl, lb = detect_zones()
        above, total = num_levels()
        
        at_support = (fh or fl) and (hb or lb)
        at_resistance = (fh or fl) and not (hb or lb)
        
        tsi_curl_bull = (tsi[i] > tsi[i-1]) and (tsi[i] < tsl[i])
        tsi_curl_bear = (tsi[i] < tsi[i-1]) and (tsi[i] > tsl[i])
        
        move_above = track_high_arr[i-1] > prev_track_high
        move_below = track_low_arr[i-1] < prev_track_low
        breakout = move_above and (above == total)
        breakdwn = move_below and (above == 0)
        
        false_break_bull = (low[i] < lowest_low_3[i-1]) and (close[i] > lowest_low_3[i-1]) and (close[i] > open_[i])
        false_break_bear = (high[i] > highest_high_3[i-1]) and (close[i] < highest_high_3[i-1]) and (close[i] < open_[i])
        
        prev_track_high = track_high_arr[i-1]
        prev_track_low = track_low_arr[i-1]
        
        track_high_arr[i] = track_high_arr[i-1]
        track_low_arr[i] = track_low_arr[i-1]
        
        for j in range(len(high_pivots)):
            if cur_close > high_levels_upper[j] and high_pivots[j] <= i:
                track_high_arr[i] += 1
            if cur_close < high_levels_lower[j] and high_pivots[j] <= i:
                track_high_arr[i] -= 1
        
        for j in range(len(low_pivots)):
            if cur_close > low_levels_upper[j] and low_pivots[j] <= i:
                track_low_arr[i] += 1
            if cur_close < low_levels_lower[j] and low_pivots[j] <= i:
                track_low_arr[i] -= 1
        
        breakout_prev = breakout
        breakdwn_prev = breakdwn
        
        long_signal = False
        short_signal = False
        
        if entry_method == "False Breaks":
            long_signal = false_break_bull and at_support
            short_signal = false_break_bear and at_resistance
        elif entry_method == "Pattern + Level":
            long_signal = at_support and bull_pattern[i] and close[i] > open_[i]
            short_signal = at_resistance and bear_pattern[i] and close[i] < open_[i]
        elif entry_method == "Confluence":
            long_signal = at_support and bull_pattern[i] and (tsi_curl_bull or not require_tsi) and close[i] > open_[i]
            short_signal = at_resistance and bear_pattern[i] and (tsi_curl_bear or not require_tsi) and close[i] < open_[i]
        else:  # Breakout Retest
            long_signal = breakout_prev and at_resistance and close[i] > open_[i] and (tsi_curl_bull or not require_tsi)
            short_signal = breakdwn_prev and at_support and close[i] < open_[i] and (tsi_curl_bear or not require_tsi)
        
        long_condition = long_signal and (total >= min_zones) and (not use_trend_filter or up_trend[i])
        short_condition = short_signal and (total >= min_zones) and (not use_trend_filter or down_trend[i])
        
        if long_condition:
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(timestamps[i]),
                'entry_time': datetime.fromtimestamp(timestamps[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close[i]),
                'raw_price_b': float(close[i])
            }
            entries.append(entry)
            trade_num += 1
        
        if short_condition:
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(timestamps[i]),
                'entry_time': datetime.fromtimestamp(timestamps[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close[i]),
                'raw_price_b': float(close[i])
            }
            entries.append(entry)
            trade_num += 1
    
    return entries