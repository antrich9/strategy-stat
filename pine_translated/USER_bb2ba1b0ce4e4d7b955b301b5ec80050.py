import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    results = []
    trade_num = 1
    
    pivotPeriod = 5
    fibLevel = 0.71
    useHTF = True
    htfLength = 50
    
    n = len(df)
    if n < pivotPeriod * 2 + 10:
        return results
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    time_vals = df['time'].values
    
    pivotHigh = np.zeros(n)
    pivotLow = np.zeros(n)
    
    for i in range(pivotPeriod, n - pivotPeriod):
        if high[i] == max(high[i-pivotPeriod:i+pivotPeriod+1]):
            pivotHigh[i] = high[i]
        if low[i] == min(low[i-pivotPeriod:i+pivotPeriod+1]):
            pivotLow[i] = low[i]
    
    if useHTF:
        htf_ema = close.ewm(span=htfLength, adjust=False).mean().values
    else:
        htf_ema = np.full(n, np.nan)
    
    majorHighLevel = np.nan
    majorLowLevel = np.nan
    majorHighIndex = -1
    majorLowIndex = -1
    externalTrend = "No Trend"
    lockBreakM = -1
    
    fibHigh = np.nan
    fibLow = np.nan
    fib071 = np.nan
    
    zigzag_types = []
    zigzag_values = []
    zigzag_indices = []
    
    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    
    for i in range(pivotPeriod + 1, n):
        current_high_pivot = pivotHigh[i] > 0
        current_low_pivot = pivotLow[i] > 0
        
        if current_high_pivot:
            high_val = high[i]
            high_idx = i
            
            if len(zigzag_types) == 0:
                zigzag_types.append("H")
                zigzag_values.append(high_val)
                zigzag_indices.append(high_idx)
            else:
                last_type = zigzag_types[-1]
                last_val = zigzag_values[-1]
                
                if "L" in last_type:
                    if high_val > last_val:
                        if len(zigzag_types) >= 2 and zigzag_values[-2] > high_val:
                            new_type = "LH"
                        else:
                            new_type = "HH"
                        zigzag_types.append(new_type)
                        zigzag_values.append(high_val)
                        zigzag_indices.append(high_idx)
                elif "H" in last_type:
                    if high_val > last_val:
                        if len(zigzag_types) >= 2 and zigzag_values[-2] > high_val:
                            new_type = "LH"
                        else:
                            new_type = "HH"
                        zigzag_types[-1] = new_type
                        zigzag_values[-1] = high_val
                        zigzag_indices[-1] = high_idx
        
        if current_low_pivot:
            low_val = low[i]
            low_idx = i
            
            if len(zigzag_types) == 0:
                zigzag_types.append("L")
                zigzag_values.append(low_val)
                zigzag_indices.append(low_idx)
            else:
                last_type = zigzag_types[-1]
                last_val = zigzag_values[-1]
                
                if "H" in last_type:
                    if low_val < last_val:
                        if len(zigzag_types) >= 2 and zigzag_values[-2] < low_val:
                            new_type = "HL"
                        else:
                            new_type = "LL"
                        zigzag_types.append(new_type)
                        zigzag_values.append(low_val)
                        zigzag_indices.append(low_idx)
                elif "L" in last_type:
                    if low_val < last_val:
                        if len(zigzag_types) >= 2 and zigzag_values[-2] < low_val:
                            new_type = "HL"
                        else:
                            new_type = "LL"
                        zigzag_types[-1] = new_type
                        zigzag_values[-1] = low_val
                        zigzag_indices[-1] = low_idx
        
        if len(zigzag_types) >= 2:
            for j in range(len(zigzag_types) - 1, -1, -1):
                t = zigzag_types[j]
                if "MHH" in t or "MLH" in t:
                    majorHighLevel = zigzag_values[j]
                    majorHighIndex = zigzag_indices[j]
                    break
                elif t.startswith("MH") and "MHL" not in t:
                    majorHighLevel = zigzag_values[j]
                    majorHighIndex = zigzag_indices[j]
                    break
            
            for j in range(len(zigzag_types) - 1, -1, -1):
                t = zigzag_types[j]
                if "MLL" in t or "MHL" in t:
                    majorLowLevel = zigzag_values[j]
                    majorLowIndex = zigzag_indices[j]
                    break
                elif t.startswith("ML") and "MLH" not in t:
                    majorLowLevel = zigzag_values[j]
                    majorLowIndex = zigzag_indices[j]
                    break
        
        if not np.isnan(majorHighLevel) and not np.isnan(majorLowLevel):
            if externalTrend == "Up Trend":
                fibLow = majorLowLevel
                fibHigh = majorHighLevel
                fib071 = fibHigh - (fibHigh - fibLow) * fibLevel
            elif externalTrend == "Down Trend":
                fibHigh = majorHighLevel
                fibLow = majorLowLevel
                fib071 = fibLow + (fibHigh - fibLow) * (1 - fibLevel)
        
        if not np.isnan(majorHighLevel) and not np.isnan(majorLowLevel):
            prev_close = close[i-1] if i > 0 else np.nan
            
            if useHTF and not np.isnan(htf_ema[i]):
                htf_bullish = close[i] > htf_ema[i]
                htf_bearish = close[i] < htf_ema[i]
            else:
                htf_bullish = True
                htf_bearish = True
            
            if i > 0 and not np.isnan(prev_close):
                if not np.isnan(majorHighLevel) and lockBreakM != majorHighIndex:
                    if prev_close <= majorHighLevel and close[i] > majorHighLevel:
                        if externalTrend == "No Trend" or externalTrend == "Up Trend":
                            externalTrend = "Up Trend"
                        elif externalTrend == "Down Trend":
                            externalTrend = "Up Trend"
                        lockBreakM = majorHighIndex
                
                if not np.isnan(majorLowLevel) and lockBreakM != majorLowIndex:
                    if prev_close >= majorLowLevel and close[i] < majorLowLevel:
                        if externalTrend == "No Trend" or externalTrend == "Down Trend":
                            externalTrend = "Down Trend"
                        elif externalTrend == "Up Trend":
                            externalTrend = "Down Trend"
                        lockBreakM = majorLowIndex
            
            if not np.isnan(fib071):
                prev_close_safe = close[i-1] if i > 0 else close[i]
                
                if externalTrend == "Up Trend" and htf_bullish:
                    if prev_close_safe >= fib071 and close[i] < fib071:
                        long_entry[i] = True
                
                elif externalTrend == "Down Trend" and htf_bearish:
                    if prev_close_safe <= fib071 and close[i] > fib071:
                        short_entry[i] = True
    
    for i in range(n):
        if long_entry[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(time_vals[i]),
                'entry_time': datetime.fromtimestamp(time_vals[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close[i]),
                'raw_price_b': float(close[i])
            })
            trade_num += 1
        elif short_entry[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(time_vals[i]),
                'entry_time': datetime.fromtimestamp(time_vals[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close[i]),
                'raw_price_b': float(close[i])
            })
            trade_num += 1
    
    return results