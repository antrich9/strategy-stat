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
    trade_num = 0
    
    bias_zone_buffer = 0.1
    max_trades_per_bias = 1
    swing_length = 5
    fvg_threshold = 0.0
    stop_loss_atr_mult = 1.5
    enable_long = True
    enable_short = True
    
    session_start = 0
    session_end = 2400
    
    htf_tf = "D"
    
    def compute_atr(df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        alpha = 1.0 / period
        for i in range(period, len(atr)):
            if not pd.isna(atr.iloc[i-1]):
                atr.iloc[i] = atr.iloc[i-1] * (1 - alpha) + tr.iloc[i] * alpha
        return atr
    
    def compute_rsi(prices, period=14):
        deltas = prices.diff()
        gains = deltas.clip(lower=0)
        losses = -deltas.clip(upper=0)
        avg_gain = gains.ewm(alpha=1.0/period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1.0/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    atr_current = compute_atr(df, 14)
    atr_htf = compute_atr(df, 14)
    auto_threshold = atr_htf * 0.1
    gap_threshold = fvg_threshold if fvg_threshold > 0 else auto_threshold
    
    htf_new = pd.Series(False, index=df.index)
    if htf_tf == "D":
        dates = pd.to_datetime(df['time'], unit='s', utc=True).dt.date
        htf_new = dates.diff().fillna(False).astype(bool)
    else:
        htf_new = pd.Series(True, index=df.index)
    
    def is_new_htf_bar(i):
        if i < 1:
            return False
        if htf_tf == "D":
            dt_curr = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc)
            dt_prev = datetime.fromtimestamp(df['time'].iloc[i-1], tz=timezone.utc)
            return dt_curr.date() != dt_prev.date()
        return True
    
    def get_pivot_high(src, length):
        result = pd.Series(np.nan, index=src.index)
        for i in range(length, len(src) - length):
            if i - length < 0:
                continue
            window = src.iloc[i-length:i+length+1]
            if window.iloc[length] == window.max():
                result.iloc[i] = src.iloc[i]
        return result
    
    def get_pivot_low(src, length):
        result = pd.Series(np.nan, index=src.index)
        for i in range(length, len(src) - length):
            if i - length < 0:
                continue
            window = src.iloc[i-length:i+length+1]
            if window.iloc[length] == window.min():
                result.iloc[i] = src.iloc[i]
        return result
    
    swing_high_htf = get_pivot_high(df['high'], swing_length)
    swing_low_htf = get_pivot_low(df['low'], swing_length)
    
    premium_levels = []
    discount_levels = []
    
    last_rejection_high = np.nan
    last_rejection_low = np.nan
    last_rejection_bar_index = -1
    
    current_bias = "neutral"
    current_bias_level = np.nan
    bias_zone_top = np.nan
    bias_zone_bottom = np.nan
    bias_start_bar = -1
    trades_in_current_bias = 0
    
    def in_session(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        minutes = dt.hour * 100 + dt.minute
        return session_start <= minutes <= session_end
    
    for i in range(len(df)):
        current_bar_index = i
        
        if is_new_htf_bar(i):
            sh_val = swing_high_htf.iloc[i] if not pd.isna(swing_high_htf.iloc[i]) else None
            sl_val = swing_low_htf.iloc[i] if not pd.isna(swing_low_htf.iloc[i]) else None
            
            if sh_val is not None:
                premium_levels.append(sh_val)
            if sl_val is not None:
                discount_levels.append(sl_val)
            
            if i >= 3:
                l_htf_1 = df['low'].iloc[i-1] if i-1 >= 0 else np.nan
                h_htf_1 = df['high'].iloc[i-1] if i-1 >= 0 else np.nan
                l_htf_3 = df['low'].iloc[i-3] if i-3 >= 0 else np.nan
                h_htf_3 = df['high'].iloc[i-3] if i-3 >= 0 else np.nan
                c_htf_2 = df['close'].iloc[i-2] if i-2 >= 0 else np.nan
                o_htf_2 = df['open'].iloc[i-2] if i-2 >= 0 else np.nan
                
                bullish_fvg_htf = (l_htf_1 > h_htf_3) and (c_htf_2 > o_htf_2)
                bearish_fvg_htf = (h_htf_1 < l_htf_3) and (c_htf_2 < o_htf_2)
                
                if bullish_fvg_htf and (l_htf_1 - h_htf_3) > gap_threshold.iloc[i] if hasattr(gap_threshold, 'iloc') else (l_htf_1 - h_htf_3) > 0.1 * atr_htf.iloc[i]:
                    discount_levels.append(l_htf_1)
                if bearish_fvg_htf and (l_htf_3 - h_htf_1) > gap_threshold.iloc[i] if hasattr(gap_threshold, 'iloc') else (l_htf_3 - h_htf_1) > 0.1 * atr_htf.iloc[i]:
                    premium_levels.append(h_htf_1)
            
            if len(premium_levels) > 200:
                premium_levels = premium_levels[-200:]
            if len(discount_levels) > 200:
                discount_levels = discount_levels[-200:]
        
        if i >= 2:
            c_htf_2 = df['close'].iloc[i-2] if i-2 >= 0 else np.nan
            o_htf_2 = df['open'].iloc[i-2] if i-2 >= 0 else np.nan
            h_htf_1 = df['high'].iloc[i-1] if i-1 >= 0 else np.nan
            h_htf_2 = df['high'].iloc[i-2] if i-2 >= 0 else np.nan
            l_htf_1 = df['low'].iloc[i-1] if i-1 >= 0 else np.nan
            l_htf_2 = df['low'].iloc[i-2] if i-2 >= 0 else np.nan
            c_htf_1 = df['close'].iloc[i-1] if i-1 >= 0 else np.nan
            o_htf_1 = df['open'].iloc[i-1] if i-1 >= 0 else np.nan
            
            bearish_rej1 = (c_htf_2 > o_htf_2) and (h_htf_2 > h_htf_1) and (c_htf_1 < o_htf_1)
            bearish_rej2 = (c_htf_2 < o_htf_2) and (h_htf_1 > h_htf_2) and (c_htf_1 < h_htf_1)
            bullish_rej1 = (c_htf_2 < o_htf_2) and (l_htf_2 < l_htf_1) and (c_htf_1 > o_htf_1)
            bullish_rej2 = (c_htf_2 > o_htf_2) and (l_htf_1 < l_htf_2) and (c_htf_1 > l_htf_1)
            
            bearish_rejection = htf_new.iloc[i] and (bearish_rej1 or bearish_rej2)
            bullish_rejection = htf_new.iloc[i] and (bullish_rej1 or bullish_rej2)
            
            if bearish_rejection:
                last_rejection_high = max(h_htf_1, h_htf_2) if not (pd.isna(h_htf_1) or pd.isna(h_htf_2)) else np.nan
                last_rejection_bar_index = current_bar_index
                if current_bias != "bearish":
                    trades_in_current_bias = 0
                current_bias = "bearish"
                current_bias_level = df['high'].iloc[i-1] if i-1 >= 0 else np.nan
                buffer_amount = current_bias_level * (bias_zone_buffer / 100) if not pd.isna(current_bias_level) else 0
                bias_zone_top = current_bias_level + buffer_amount if not pd.isna(current_bias_level) else np.nan
                bias_zone_bottom = current_bias_level - buffer_amount if not pd.isna(current_bias_level) else np.nan
                bias_start_bar = current_bar_index
            
            if bullish_rejection:
                last_rejection_low = min(l_htf_1, l_htf_2) if not (pd.isna(l_htf_1) or pd.isna(l_htf_2)) else np.nan
                last_rejection_bar_index = current_bar_index
                if current_bias != "bullish":
                    trades_in_current_bias = 0
                current_bias = "bullish"
                current_bias_level = df['low'].iloc[i-1] if i-1 >= 0 else np.nan
                buffer_amount = current_bias_level * (bias_zone_buffer / 100) if not pd.isna(current_bias_level) else 0
                bias_zone_top = current_bias_level + buffer_amount if not pd.isna(current_bias_level) else np.nan
                bias_zone_bottom = current_bias_level - buffer_amount if not pd.isna(current_bias_level) else np.nan
                bias_start_bar = current_bar_index
        
        close_current = df['close'].iloc[i]
        high_current = df['high'].iloc[i]
        low_current = df['low'].iloc[i]
        ts_current = df['time'].iloc[i]
        
        sweep_up = (not pd.isna(last_rejection_high) and last_rejection_bar_index >= 0 and 
                    current_bar_index > last_rejection_bar_index and high_current > last_rejection_high)
        sweep_down = (not pd.isna(last_rejection_low) and last_rejection_bar_index >= 0 and 
                      current_bar_index > last_rejection_bar_index and low_current < last_rejection_low)
        
        target_up = np.nan
        target_dn = np.nan
        
        if sweep_up and not pd.isna(last_rejection_high):
            candidates = [p for p in premium_levels if not pd.isna(p) and p > last_rejection_high]
            if candidates:
                target_up = min(candidates)
        
        if sweep_down and not pd.isna(last_rejection_low):
            candidates = [d for d in discount_levels if not pd.isna(d) and d < last_rejection_low]
            if candidates:
                target_dn = max(candidates)
        
        in_bullish_bias_zone = (current_bias == "bullish" and not pd.isna(bias_zone_top) and 
                                not pd.isna(bias_zone_bottom) and 
                                bias_zone_bottom <= close_current <= bias_zone_top)
        in_bearish_bias_zone = (current_bias == "bearish" and not pd.isna(bias_zone_top) and 
                                not pd.isna(bias_zone_bottom) and 
                                bias_zone_bottom <= close_current <= bias_zone_top)
        
        in_session_val = in_session(ts_current)
        
        long_condition = (enable_long and in_session_val and current_bias == "bullish" and 
                         in_bullish_bias_zone and sweep_up and not pd.isna(target_up) and 
                         trades_in_current_bias < max_trades_per_bias)
        
        short_condition = (enable_short and in_session_val and current_bias == "bearish" and 
                          in_bearish_bias_zone and sweep_down and not pd.isna(target_dn) and 
                          trades_in_current_bias < max_trades_per_bias)
        
        if long_condition:
            trade_num += 1
            entry_time_str = datetime.fromtimestamp(ts_current, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts_current),
                'entry_time': entry_time_str,
                'entry_price_guess': float(close_current),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_current),
                'raw_price_b': float(close_current)
            })
            trades_in_current_bias += 1
        
        if short_condition:
            trade_num += 1
            entry_time_str = datetime.fromtimestamp(ts_current, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts_current),
                'entry_time': entry_time_str,
                'entry_price_guess': float(close_current),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_current),
                'raw_price_b': float(close_current)
            })
            trades_in_current_bias += 1
    
    return entries