import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    PP = 5
    ATR_PERIOD = 55
    ATR_THRESHOLD_MULT = 1.5
    MIN_BARS_BETWEEN_SIGNALS = 10
    SKIP_WARMUP_BARS = 30
    
    entries = []
    trade_num = 0
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    ts = df['time'].values
    
    n = len(df)
    
    prev_tr = np.abs(np.diff(high))[:-1]
    prev_tr = np.insert(prev_tr, 0, high[0] - low[0])
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    
    atr = np.zeros(n)
    if n > ATR_PERIOD:
        sma = np.mean(tr[:ATR_PERIOD])
        atr[ATR_PERIOD - 1] = sma
        for i in range(ATR_PERIOD, n):
            atr[i] = (atr[i-1] * (ATR_PERIOD - 1) + tr[i]) / ATR_PERIOD
    
    major_high = np.nan
    major_low = np.nan
    major_high_idx = -1
    major_low_idx = -1
    
    major_bull_signal = False
    major_bear_signal = False
    last_major_bull_idx = -1
    last_major_bear_idx = -1
    
    entry_idx = -1
    entry_price = np.nan
    entry_direction = ''
    
    for i in range(PP, n - PP):
        if np.isnan(atr[i]):
            continue
        
        is_high_pivot = True
        is_low_pivot = False
        
        for j in range(1, PP + 1):
            if high[i] <= high[i + j] or high[i] <= high[i - j]:
                is_high_pivot = False
                break
        
        if not is_high_pivot:
            for j in range(1, PP + 1):
                if low[i] >= low[i + j] or low[i] >= low[i - j]:
                    is_low_pivot = False
                    break
        
        if is_high_pivot:
            new_major_high = not np.isnan(major_high) and high[i] > major_high or np.isnan(major_high)
            if new_major_high and not np.isnan(major_high):
                prev_idx = major_high_idx
                if prev_idx >= 0 and prev_idx >= 1:
                    if high[i] > high[prev_idx]:
                        major_bull_signal = True
                        last_major_bull_idx = i
            major_high = high[i]
            major_high_idx = i
        
        elif is_low_pivot:
            new_major_low = not np.isnan(major_low) and low[i] < major_low or np.isnan(major_low)
            if new_major_low and not np.isnan(major_low):
                prev_idx = major_low_idx
                if prev_idx >= 0 and prev_idx >= 1:
                    if low[i] < low[prev_idx]:
                        major_bear_signal = True
                        last_major_bear_idx = i
            major_low = low[i]
            major_low_idx = i
        
        bull_break = (last_major_bull_idx >= 0 and 
                      major_high_idx >= 0 and
                      high[major_high_idx] > high[last_major_bull_idx])
        bear_break = (last_major_bear_idx >= 0 and 
                      major_low_idx >= 0 and
                      low[major_low_idx] < low[last_major_bear_idx])
        
        if bull_break and (entry_idx < 0 or i - entry_idx > MIN_BARS_BETWEEN_SIGNALS):
            if i >= SKIP_WARMUP_BARS:
                trade_num += 1
                entry_idx = i
                entry_price = close[i]
                entry_direction = 'long'
                major_bull_signal = False
                major_bear_signal = False
        
        elif bear_break and (entry_idx < 0 or i - entry_idx > MIN_BARS_BETWEEN_SIGNALS):
            if i >= SKIP_WARMUP_BARS:
                trade_num += 1
                entry_idx = i
                entry_price = close[i]
                entry_direction = 'short'
                major_bull_signal = False
                major_bear_signal = False
        
        elif entry_idx >= 0:
            if entry_direction == 'long':
                price_change = entry_price - close[i]
                if price_change > ATR_THRESHOLD_MULT * atr[i]:
                    entry_ts = int(ts[entry_idx])
                    exit_ts = int(ts[i])
                    entry_time_str = datetime.fromtimestamp(entry_ts // 1000 if entry_ts > 1e12 else entry_ts, tz=timezone.utc).isoformat()
                    exit_time_str = datetime.fromtimestamp(exit_ts // 1000 if exit_ts > 1e12 else exit_ts, tz=timezone.utc).isoformat()
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': entry_ts,
                        'entry_time': entry_time_str,
                        'entry_price_guess': entry_price,
                        'exit_ts': exit_ts,
                        'exit_time': exit_time_str,
                        'exit_price_guess': close[i],
                        'raw_price_a': entry_price,
                        'raw_price_b': entry_price
                    })
                    entry_idx = -1
                    entry_price = np.nan
                    entry_direction = ''
                    major_bull_signal = False
                    major_bear_signal = False
            elif entry_direction == 'short':
                price_change = close[i] - entry_price
                if price_change > ATR_THRESHOLD_MULT * atr[i]:
                    entry_ts = int(ts[entry_idx])
                    exit_ts = int(ts[i])
                    entry_time_str = datetime.fromtimestamp(entry_ts // 1000 if entry_ts > 1e12 else entry_ts, tz=timezone.utc).isoformat()
                    exit_time_str = datetime.fromtimestamp(exit_ts // 1000 if exit_ts > 1e12 else exit_ts, tz=timezone.utc).isoformat()
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': entry_ts,
                        'entry_time': entry_time_str,
                        'entry_price_guess': entry_price,
                        'exit_ts': exit_ts,
                        'exit_time': exit_time_str,
                        'exit_price_guess': close[i],
                        'raw_price_a': entry_price,
                        'raw_price_b': entry_price
                    })
                    entry_idx = -1
                    entry_price = np.nan
                    entry_direction = ''
                    major_bull_signal = False
                    major_bear_signal = False
    
    return entries