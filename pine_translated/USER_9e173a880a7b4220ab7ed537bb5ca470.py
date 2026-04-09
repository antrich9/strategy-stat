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
    entries = []
    trade_num = 1
    
    if len(df) < 50:
        return entries
    
    o = df['open'].copy()
    h = df['high'].copy()
    l = df['low'].copy()
    c = df['close'].copy()
    t = df['time'].copy()
    
    # Default settings from Pine Script
    entry_method = "Confluence"
    use_trend_filter = True
    trend_length = 50
    min_zones = 2
    require_tsi = False
    left = 20
    right = 15
    n_piv = 4
    atr_len = 30
    mult = 0.5
    per = 5
    src_zone = "HA"
    shortf = 5
    longf = 25
    signalf = 14
    
    # ATR (Wilder)
    tr = np.maximum(np.maximum(h - l, np.abs(h - c.shift(1))), np.abs(l - c.shift(1)))
    tr.iloc[0] = h.iloc[0] - l.iloc[0]
    atr = tr.ewm(alpha=1/atr_len, adjust=False).mean()
    
    # Heikin Ashi
    ha_close = (o + h + l + c) / 4
    ha_open = pd.Series(index=o.index, dtype=float)
    ha_open.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2
    for i in range(1, len(ha_open)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    
    hi_ha_bod = pd.concat([ha_close, ha_open], axis=1).max(axis=1)
    lo_ha_bod = pd.concat([ha_close, ha_open], axis=1).min(axis=1)
    hi_bod = pd.concat([o, c], axis=1).max(axis=1)
    lo_bod = pd.concat([o, c], axis=1).min(axis=1)
    
    src_high = hi_ha_bod if src_zone == "HA" else (h if src_zone == "High/Low" else hi_bod)
    src_low = lo_ha_bod if src_zone == "HA" else (l if src_zone == "High/Low" else lo_bod)
    
    # Pivot detection
    perc = c * (per / 100)
    band = pd.Series(index=c.index, dtype=float)
    for i in range(len(c)):
        if i >= right:
            val = min(atr.iloc[i] * mult, perc.iloc[i])
            band.iloc[i] = val / 2
        else:
            band.iloc[i] = np.nan
    
    # Pivot high/low with zone tracking
    pivot_high_arr = []
    pivot_low_arr = []
    high_bull_arr = []
    low_bull_arr = []
    track_high = pd.Series(0, index=c.index)
    track_lows = pd.Series(0, index=c.index)
    
    for i in range(left + right, len(c)):
        ph_found = False
        pl_found = False
        for j in range(left):
            if i - j - right > 0:
                if src_high.iloc[i - j - right] <= src_high.iloc[i - right] and all(src_high.iloc[i - j - right + k] <= src_high.iloc[i - right + k] for k in range(1, j + 1) if i - j - right + k <= i - right):
                    ph_idx = i - right
                    ph_val = src_high.iloc[ph_idx]
                    ph_found = True
                    break
        for j in range(left):
            if i - j - right > 0:
                if src_low.iloc[i - j - right] >= src_low.iloc[i - right] and all(src_low.iloc[i - j - right + k] >= src_low.iloc[i - right + k] for k in range(1, j + 1) if i - j - right + k <= i - right):
                    pl_idx = i - right
                    pl_val = src_low.iloc[pl_idx]
                    pl_found = True
                    break
        
        if ph_found:
            pivot_high_arr.insert(0, (i - right, ph_val + band.iloc[i - right] if not np.isnan(band.iloc[i - right]) else ph_val, ph_val - band.iloc[i - right] if not np.isnan(band.iloc[i - right]) else ph_val))
            high_bull_arr.insert(0, False)
            if len(pivot_high_arr) > n_piv:
                pivot_high_arr.pop()
                high_bull_arr.pop()
        if pl_found:
            pivot_low_arr.insert(0, (i - right, pl_val + band.iloc[i - right] if not np.isnan(band.iloc[i - right]) else pl_val, pl_val - band.iloc[i - right] if not np.isnan(band.iloc[i - right]) else pl_val))
            low_bull_arr.insert(0, True)
            if len(pivot_low_arr) > n_piv:
                pivot_low_arr.pop()
                low_bull_arr.pop()
        
        # Zone detection
        at_support = False
        at_resistance = False
        above = 0
        total = 0
        
        for idx, (ph_idx, ph_top, ph_bot) in enumerate(pivot_high_arr):
            if c.iloc[i] > ph_top:
                high_bull_arr[idx] = True
            if c.iloc[i] < ph_bot:
                high_bull_arr[idx] = False
            if high_bull_arr[idx]:
                above += 1
            total += 1
        
        for idx, (pl_idx, pl_top, pl_bot) in enumerate(pivot_low_arr):
            if c.iloc[i] > pl_top:
                low_bull_arr[idx] = True
            if c.iloc[i] < pl_bot:
                low_bull_arr[idx] = False
            if low_bull_arr[idx]:
                above += 1
            total += 1
        
        # Detect if at support/resistance
        fh = fl = False
        hb = lb = True
        for idx, (ph_idx, ph_top, ph_bot) in enumerate(pivot_high_arr):
            if l.iloc[i] < ph_top and h.iloc[i] > ph_bot:
                fh = True
                hb = high_bull_arr[idx]
                break
        for idx, (pl_idx, pl_top, pl_bot) in enumerate(pivot_low_arr):
            if l.iloc[i] < pl_top and h.iloc[i] > pl_bot:
                fl = True
                lb = low_bull_arr[idx]
                break
        
        at_support = (fh or fl) and (hb or lb)
        at_resistance = (fh or fl) and not (hb or lb)
        
        track_high.iloc[i] = len([b for b in high_bull_arr if b])
        track_lows.iloc[i] = len([b for b in low_bull_arr if b])
        
        # TSI calculation
        if i >= 1:
            diff = c.diff().fillna(0)
            long_ema = diff.ewm(span=longf, adjust=False).mean()
            short_ema = long_ema.ewm(span=shortf, adjust=False).mean()
            tsi = (short_ema / long_ema.abs()) * 100 if long_ema.abs().iloc[i] != 0 else 0
            tsl_val = tsi.ewm(span=signalf, adjust=False).mean() if i >= signalf else 0
            tsi_curl_bull = tsi.iloc[i] > tsi.iloc[i-1] if i >= 1 else False and tsi.iloc[i] < tsl_val if i >= signalf else False
            tsi_curl_bear = tsi.iloc[i] < tsi.iloc[i-1] if i >= 1 else False and tsi.iloc[i] > tsl_val if i >= signalf else False
        else:
            tsi_curl_bull = tsi_curl_bear = False
        
        # Trend filter
        trend_ema = c.ewm(span=trend_length, adjust=False).mean()
        up_trend = c.iloc[i] > trend_ema.iloc[i]
        down_trend = c.iloc[i] < trend_ema.iloc[i]
        
        # False breaks
        lowest_3_low = l.rolling(3).min()
        highest_3_high = h.rolling(3).max()
        
        false_break_bull = (l.iloc[i] < lowest_3_low.iloc[i-1] if i >= 1 else False) and c.iloc[i] > lowest_3_low.iloc[i-1] if i >= 1 else False and c.iloc[i] > o.iloc[i]
        false_break_bear = (h.iloc[i] > highest_3_high.iloc[i-1] if i >= 1 else False) and c.iloc[i] < highest_3_high.iloc[i-1] if i >= 1 else False and c.iloc[i] < o.iloc[i]
        
        # Breakout/breakdown
        move_above = track_high.iloc[i] > track_high.iloc[i-1] if i >= 1 else False
        move_below = track_lows.iloc[i] < track_lows.iloc[i-1] if i >= 1 else False
        break_out = move_above and above == total
        break_dwn = move_below and above == 0
        
        # Pattern detection (simplified - treating as always true for conditions)
        bull_pattern = True
        bear_pattern = True
        
        # Entry conditions
        long_false_break = false_break_bull and at_support
        short_false_break = false_break_bear and at_resistance
        long_pattern = at_support and bull_pattern and c.iloc[i] > o.iloc[i]
        short_pattern = at_resistance and bear_pattern and c.iloc[i] < o.iloc[i]
        long_confluence = at_support and bull_pattern and (tsi_curl_bull or not require_tsi) and c.iloc[i] > o.iloc[i]
        short_confluence = at_resistance and bear_pattern and (tsi_curl_bear or not require_tsi) and c.iloc[i] < o.iloc[i]
        long_retest = break_out and at_resistance and c.iloc[i] > o.iloc[i] and (tsi_curl_bull or not require_tsi)
        short_retest = break_dwn and at_support and c.iloc[i] < o.iloc[i] and (tsi_curl_bear or not require_tsi)
        
        if entry_method == "False Breaks":
            long_signal = long_false_break
            short_signal = short_false_break
        elif entry_method == "Pattern + Level":
            long_signal = long_pattern
            short_signal = short_pattern
        elif entry_method == "Confluence":
            long_signal = long_confluence
            short_signal = short_confluence
        else:
            long_signal = long_retest
            short_signal = short_retest
        
        long_condition = long_signal and total >= min_zones and (not use_trend_filter or up_trend)
        short_condition = short_signal and total >= min_zones and (not use_trend_filter or down_trend)
        
        if long_condition:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(t.iloc[i]),
                'entry_time': datetime.fromtimestamp(t.iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(c.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(c.iloc[i]),
                'raw_price_b': float(c.iloc[i])
            })
            trade_num += 1
        
        if short_condition:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(t.iloc[i]),
                'entry_time': datetime.fromtimestamp(t.iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(c.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(c.iloc[i]),
                'raw_price_b': float(c.iloc[i])
            })
            trade_num += 1
    
    return entries