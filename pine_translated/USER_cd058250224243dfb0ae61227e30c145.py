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
    # Default parameters from Pine Script
    atrLength = 14
    lengthMD = 10
    tradeDirection = "Both"
    
    # Precision Trend Histogram parameters
    lengthShort = 5
    lengthLong = 21
    
    # S&R parameters
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    input_repType = 'On'
    
    rTon = input_repType == 'On'
    
    n = len(df)
    if n < input_lookback + 2:
        return []
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Precision Trend Histogram
    src = close
    maShort = src.ewm(span=lengthShort, adjust=False).mean()
    maLong = src.ewm(span=lengthLong, adjust=False).mean()
    precisionTrend = maShort - maLong
    
    # McGinley Dynamic implementation
    md = pd.Series(index=df.index, dtype=float)
    md.iloc[0] = close.iloc[0]
    for i in range(1, n):
        prev_md = md.iloc[i-1]
        prev_close = close.iloc[i-1]
        curr_close = close.iloc[i]
        k = 0.6 * (curr_close - prev_close) / (prev_close * 0.1)
        md.iloc[i] = prev_md + (curr_close - prev_md) / (k * lengthMD * (prev_md / curr_close) ** 4 if prev_md != 0 else 1)
    
    # Wilder RSI implementation
    def wilder_rsi(src, length):
        delta = src.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilder ATR implementation
    def wilder_atr(high, low, close, length):
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/length, adjust=False).mean()
        return atr
    
    atr = wilder_atr(high, low, close, atrLength)
    rsi_md = wilder_rsi(close, lengthMD)
    
    # Pivot points
    bb = input_lookback
    
    def pivot_low_series(low, length):
        result = pd.Series(np.nan, index=low.index)
        for i in range(length, n - length):
            is_pivot = True
            for j in range(1, length + 1):
                if low.iloc[i - j] <= low.iloc[i] or low.iloc[i + j] < low.iloc[i]:
                    is_pivot = False
                    break
            if is_pivot:
                result.iloc[i] = low.iloc[i]
        return result
    
    def pivot_high_series(high, length):
        result = pd.Series(np.nan, index=high.index)
        for i in range(length, n - length):
            is_pivot = True
            for j in range(1, length + 1):
                if high.iloc[i - j] >= high.iloc[i] or high.iloc[i + j] > high.iloc[i]:
                    is_pivot = False
                    break
            if is_pivot:
                result.iloc[i] = high.iloc[i]
        return result
    
    pl = pivot_low_series(low, bb)
    ph = pivot_high_series(high, bb)
    pl = pl.ffill()
    ph = ph.ffill()
    
    change_pl = pl.diff() != 0
    change_ph = ph.diff() != 0
    
    # Box boundaries
    sTop_arr = pd.Series(np.nan, index=df.index)
    sBot_arr = pd.Series(np.nan, index=df.index)
    rTop_arr = pd.Series(np.nan, index=df.index)
    rBot_arr = pd.Series(np.nan, index=df.index)
    
    for i in range(bb + 1, n):
        if change_pl.iloc[i]:
            s_yLoc = low.iloc[bb + 1] if low.iloc[bb + 1] > low.iloc[bb - 1] else low.iloc[bb - 1]
            sTop_arr.iloc[i] = pl.iloc[i]
            sBot_arr.iloc[i] = s_yLoc
        else:
            sTop_arr.iloc[i] = sTop_arr.iloc[i-1]
            sBot_arr.iloc[i] = sBot_arr.iloc[i-1]
        
        if change_ph.iloc[i]:
            r_yLoc = high.iloc[bb + 1] if high.iloc[bb + 1] > high.iloc[bb - 1] else high.iloc[bb - 1]
            rBot_arr.iloc[i] = ph.iloc[i]
            rTop_arr.iloc[i] = r_yLoc
        else:
            rTop_arr.iloc[i] = rTop_arr.iloc[i-1]
            rBot_arr.iloc[i] = rBot_arr.iloc[i-1]
    
    sTop = sTop_arr
    sBot = sBot_arr
    rTop = rTop_arr
    rBot = rBot_arr
    
    # Breakout conditions
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)
    
    co = pd.Series(False, index=df.index)
    cu = pd.Series(False, index=df.index)
    
    for i in range(bb + 1, n):
        sBot_val = sBot.iloc[i]
        rTop_val = rTop.iloc[i]
        
        if rTon:
            co.iloc[i] = close.iloc[i] > rTop_val and close.iloc[i-1] <= rTop_val
            cu.iloc[i] = close.iloc[i] < sBot_val and close.iloc[i-1] >= sBot_val
        else:
            co.iloc[i] = high.iloc[i] > rTop_val and high.iloc[i-1] <= rTop_val
            cu.iloc[i] = low.iloc[i] < sBot_val and low.iloc[i-1] >= sBot_val
    
    # Update sBreak and rBreak
    for i in range(bb + 1, n):
        if cu.iloc[i] and pd.isna(sBreak.iloc[i-1]):
            sBreak.iloc[i] = True
        elif change_pl.iloc[i]:
            sBreak.iloc[i] = False
        else:
            sBreak.iloc[i] = sBreak.iloc[i-1]
        
        if co.iloc[i] and pd.isna(rBreak.iloc[i-1]):
            rBreak.iloc[i] = True
        elif change_ph.iloc[i]:
            rBreak.iloc[i] = False
        else:
            rBreak.iloc[i] = rBreak.iloc[i-1]
    
    # Retest conditions for support
    sRetValid = pd.Series(False, index=df.index)
    sRetEvent = pd.Series(False, index=df.index)
    
    for i in range(bb + 1, n):
        if sBreak.iloc[i]:
            bars_since_break = 0
            for j in range(i-1, -1, -1):
                if sBreak.iloc[j] and not pd.isna(sBreak.iloc[j]):
                    bars_since_break = i - j
                    break
            
            if bars_since_break > input_retSince:
                sTop_val = sTop.iloc[i]
                sBot_val = sBot.iloc[i]
                
                cond1 = high.iloc[i] >= sTop_val and close.iloc[i] <= sBot_val
                cond2 = high.iloc[i] >= sTop_val and close.iloc[i] >= sBot_val and close.iloc[i] <= sTop_val
                cond3 = high.iloc[i] >= sBot_val and high.iloc[i] <= sTop_val
                cond4 = high.iloc[i] >= sBot_val and high.iloc[i] <= sTop_val and close.iloc[i] < sBot_val
                
                retActive = cond1 or cond2 or cond3 or cond4
                
                if retActive and (i == 0 or not sRetEvent.iloc[i-1]):
                    sRetEvent.iloc[i] = True
                
                bars_since_event = 0
                if sRetEvent.iloc[i]:
                    for j in range(i-1, -1, -1):
                        if sRetEvent.iloc[j]:
                            bars_since_event = i - j
                            break
                    
                    if bars_since_event > 0 and bars_since_event <= input_retValid:
                        sRetValid.iloc[i] = True
    
    # Retest conditions for resistance
    rRetValid = pd.Series(False, index=df.index)
    rRetEvent = pd.Series(False, index=df.index)
    
    for i in range(bb + 1, n):
        if rBreak.iloc[i]:
            bars_since_break = 0
            for j in range(i-1, -1, -1):
                if rBreak.iloc[j] and not pd.isna(rBreak.iloc[j]):
                    bars_since_break = i - j
                    break
            
            if bars_since_break > input_retSince:
                rTop_val = rTop.iloc[i]
                rBot_val = rBot.iloc[i]
                
                cond1 = low.iloc[i] <= rBot_val and close.iloc[i] >= rTop_val
                cond2 = low.iloc[i] <= rBot_val and close.iloc[i] <= rTop_val and close.iloc[i] >= rBot_val
                cond3 = low.iloc[i] <= rTop_val and low.iloc[i] >= rBot_val
                cond4 = low.iloc[i] <= rTop_val and low.iloc[i] >= rBot_val and close.iloc[i] > rTop_val
                
                retActive = cond1 or cond2 or cond3 or cond4
                
                if retActive and (i == 0 or not rRetEvent.iloc[i-1]):
                    rRetEvent.iloc[i] = True
                
                bars_since_event = 0
                if rRetEvent.iloc[i]:
                    for j in range(i-1, -1, -1):
                        if rRetEvent.iloc[j]:
                            bars_since_event = i - j
                            break
                    
                    if bars_since_event > 0 and bars_since_event <= input_retValid:
                        rRetValid.iloc[i] = True
    
    # Determine trade conditions
    longCondition = tradeDirection in ["Long", "Both"]
    shortCondition = tradeDirection in ["Short", "Both"]
    
    # Build entry conditions
    long_entry_cond = sRetValid & longCondition
    short_entry_cond = rRetValid & shortCondition
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(n):
        if np.isnan(atr.iloc[i]) or np.isnan(precisionTrend.iloc[i]):
            continue
        
        entry_price = close.iloc[i]
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        if long_entry_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if short_entry_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries