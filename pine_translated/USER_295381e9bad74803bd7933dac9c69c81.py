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
    
    enable_long = True
    enable_short = True
    bias_zone_buffer = 0.1
    max_trades_per_bias = 1
    swing_length = 5
    gap_threshold = 0.0
    
    def notna(val):
        return val == val and val is not None
    
    entries = []
    trade_num = 1
    
    premium_levels = []
    discount_levels = []
    current_bias = "neutral"
    bias_zone_top = np.nan
    bias_zone_bottom = np.nan
    last_rejection_high = np.nan
    last_rejection_low = np.nan
    last_rejection_bar_index = -1
    trades_in_bias = 0
    
    htf_tf = "D"
    
    prev_day = None
    
    for i in range(len(df)):
        ts = df['time'].iloc[i]
        current_day = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        
        htf_new = (prev_day is not None and current_day != prev_day) or (i == 0 and prev_day is None)
        prev_day = current_day
        
        swing_high_val = np.nan
        swing_low_val = np.nan
        if i >= swing_length * 2:
            window_high = df['high'].iloc[i - swing_length:i]
            max_idx = window_high.idxmax()
            if max_idx >= swing_length and max_idx < len(df['high'].iloc[:i]):
                is_pivot_high = True
                for j in range(max_idx - swing_length, max_idx):
                    if j >= 0 and df['high'].iloc[j] >= df['high'].iloc[max_idx]:
                        is_pivot_high = False
                        break
                for j in range(max_idx + 1, max_idx + swing_length + 1):
                    if j < len(df['high'].iloc[:i]) and df['high'].iloc[j] >= df['high'].iloc[max_idx]:
                        is_pivot_high = False
                        break
                if is_pivot_high:
                    swing_high_val = df['high'].iloc[max_idx]
            
            window_low = df['low'].iloc[i - swing_length:i]
            min_idx = window_low.idxmin()
            if min_idx >= swing_length and min_idx < len(df['low'].iloc[:i]):
                is_pivot_low = True
                for j in range(min_idx - swing_length, min_idx):
                    if j >= 0 and df['low'].iloc[j] <= df['low'].iloc[min_idx]:
                        is_pivot_low = False
                        break
                for j in range(min_idx + 1, min_idx + swing_length + 1):
                    if j < len(df['low'].iloc[:i]) and df['low'].iloc[j] <= df['low'].iloc[min_idx]:
                        is_pivot_low = False
                        break
                if is_pivot_low:
                    swing_low_val = df['low'].iloc[min_idx]
        
        if htf_new:
            if notna(swing_high_val):
                premium_levels.append(swing_high_val)
            if notna(swing_low_val):
                discount_levels.append(swing_low_val)
            
            bull_fvg = False
            bear_fvg = False
            bull_fvg_top = np.nan
            bull_fvg_bottom = np.nan
            bear_fvg_top = np.nan
            bear_fvg_bottom = np.nan
            
            if i >= 3:
                bull_fvg = (df['low'].iloc[i-1] > df['high'].iloc[i-3]) and (df['close'].iloc[i-2] > df['open'].iloc[i-2])
                bear_fvg = (df['high'].iloc[i-1] < df['low'].iloc[i-3]) and (df['close'].iloc[i-2] < df['open'].iloc[i-2])
                bull_fvg_top = df['low'].iloc[i-1]
                bull_fvg_bottom = df['high'].iloc[i-3]
                bear_fvg_top = df['low'].iloc[i-3]
                bear_fvg_bottom = df['high'].iloc[i-1]
            
            auto_thresh = 0.0
            thresh = gap_threshold if gap_threshold > 0 else auto_thresh
            
            if bull_fvg and ((bull_fvg_top - bull_fvg_bottom) > thresh):
                discount_levels.append(bull_fvg_top)
            if bear_fvg and ((bear_fvg_top - bear_fvg_bottom) > thresh):
                premium_levels.append(bear_fvg_bottom)
            
            if len(premium_levels) > 200:
                premium_levels = premium_levels[-200:]
            if len(discount_levels) > 200:
                discount_levels = discount_levels[-200:]
        
        bear_rej1 = False
        bear_rej2 = False
        bull_rej1 = False
        bull_rej2 = False
        
        if i >= 2:
            bear_rej1 = (df['close'].iloc[i-2] > df['open'].iloc[i-2]) and (df['high'].iloc[i-2] > df['high'].iloc[i-1]) and (df['close'].iloc[i-1] < df['open'].iloc[i-1])
            bear_rej2 = (df['close'].iloc[i-2] < df['open'].iloc[i-2]) and (df['high'].iloc[i-1] > df['high'].iloc[i-2]) and (df['close'].iloc[i-1] < df['high'].iloc[i-1])
            bull_rej1 = (df['close'].iloc[i-2] < df['open'].iloc[i-2]) and (df['low'].iloc[i-2] < df['low'].iloc[i-1]) and (df['close'].iloc[i-1] > df['open'].iloc[i-1])
            bull_rej2 = (df['close'].iloc[i-2] > df['open'].iloc[i-2]) and (df['low'].iloc[i-1] < df['low'].iloc[i-2]) and (df['close'].iloc[i-1] > df['low'].iloc[i-1])
        
        bearish_rejection = htf_new and (bear_rej1 or bear_rej2)
        bullish_rejection = htf_new and (bull_rej1 or bull_rej2)
        
        if bearish_rejection:
            last_rejection_high = max(df['high'].iloc[i-1], df['high'].iloc[i-2]) if i >= 2 else df['high'].iloc[i]
            last_rejection_bar_index = i
            if current_bias != "bearish":
                trades_in_bias = 0
            current_bias = "bearish"
            bias_level = df['high'].iloc[i-1]
            buffer_amt = bias_level * (bias_zone_buffer / 100)
            bias_zone_top = bias_level + buffer_amt
            bias_zone_bottom = bias_level - buffer_amt
        
        if bullish_rejection:
            last_rejection_low = min(df['low'].iloc[i-1], df['low'].iloc[i-2]) if i >= 2 else df['low'].iloc[i]
            last_rejection_bar_index = i
            if current_bias != "bullish":
                trades_in_bias = 0
            current_bias = "bullish"
            bias_level = df['low'].iloc[i-1]
            buffer_amt = bias_level * (bias_zone_buffer / 100)
            bias_zone_top = bias_level + buffer_amt
            bias_zone_bottom = bias_level - buffer_amt
        
        in_bull_zone = current_bias == "bullish" and notna(bias_zone_top) and notna(bias_zone_bottom) and df['close'].iloc[i] >= bias_zone_bottom and df['close'].iloc[i] <= bias_zone_top
        in_bear_zone = current_bias == "bearish" and notna(bias_zone_top) and notna(bias_zone_bottom) and df['close'].iloc[i] >= bias_zone_bottom and df['close'].iloc[i] <= bias_zone_top
        
        sweep_up = notna(last_rejection_high) and i > last_rejection_bar_index and df['high'].iloc[i] > last_rejection_high
        sweep_down = notna(last_rejection_low) and i > last_rejection_bar_index and df['low'].iloc[i] < last_rejection_low
        
        target_up = np.nan
        target_dn = np.nan
        
        if sweep_up:
            target_up = np.nan
            for val in premium_levels:
                if notna(val) and val > last_rejection_high:
                    target_up = val if notna(target_up) else target_up
                    if val < target_up:
                        target_up = val
        
        if sweep_down:
            target_dn = np.nan
            for val in discount_levels:
                if notna(val) and val < last_rejection_low:
                    target_dn = val if notna(target_dn) else val
                    if val > target_dn:
                        target_dn = val
        
        long_condition = enable_long and in_bull_zone and sweep_up and notna(target_up) and trades_in_bias < max_trades_per_bias
        short_condition = enable_short and in_bear_zone and sweep_down and notna(target_dn) and trades_in_bias < max_trades_per_bias
        
        if long_condition:
            entry_price = df['close'].iloc[i]
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            trades_in_bias += 1
        
        if short_condition:
            entry_price = df['close'].iloc[i]
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            trades_in_bias += 1
    
    return entries