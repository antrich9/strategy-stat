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
    
    # Parameters
    atrLength = 14
    atrMultiplier = 1.5
    takeProfitRatio = 1.5
    tradeDirection = "Both"
    kalmanLength = 3
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    input_breakout = True
    input_retest = True
    input_repType = 'On'
    
    bb = input_lookback
    
    rTon = input_repType == 'On'
    rTcc = input_repType == 'Off: Candle Confirmation'
    rThv = input_repType == 'Off: High & Low'
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate ATR using Wilder's method
    tr = np.maximum(
        np.maximum(
            high - low,
            np.abs(high - close.shift(1))
        ),
        np.abs(low - close.shift(1))
    )
    
    atr = tr.ewm(alpha=1/atrLength, adjust=False).mean()
    
    stopLossLong = close - (atr * atrMultiplier)
    stopLossShort = close + (atr * atrMultiplier)
    takeProfitLong = close + ((atr * atrMultiplier) * takeProfitRatio)
    takeProfitShort = close - ((atr * atrMultiplier) * takeProfitRatio)
    
    # Pivot calculations
    pl = low.rolling(window=bb+1).min().shift(1)
    ph = high.rolling(window=bb+1).max().shift(1)
    
    pl = pl.where(low.shift(bb) > low.shift(bb + 1), low.shift(bb - 1))
    ph = ph.where(high.shift(bb) > high.shift(bb + 1), high.shift(bb - 1))
    
    pl = pl.fillna(method='ffill')
    ph = ph.fillna(method='ffill')
    
    # Box locations
    s_yLoc = np.where(low.shift(bb + 1) > low.shift(bb - 1), low.shift(bb - 1), low.shift(bb + 1))
    r_yLoc = np.where(high.shift(bb + 1) > high.shift(bb - 1), high.shift(bb + 1), high.shift(bb - 1))
    
    # Initialize breakout flags
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)
    
    # Retest valid series
    sRetValid = pd.Series(False, index=df.index)
    rRetValid = pd.Series(False, index=df.index)
    
    retOccurred_s = pd.Series(False, index=df.index)
    retOccurred_r = pd.Series(False, index=df.index)
    
    # Process bars
    for i in range(bb + 2, len(df)):
        if pd.isna(atr.iloc[i]) or pd.isna(pl.iloc[i]) or pd.isna(ph.iloc[i]):
            continue
            
        sTop_val = s_yLoc[i] if not pd.isna(s_yLoc[i]) else low.iloc[i]
        sBot_val = pl.iloc[i] if not pd.isna(pl.iloc[i]) else low.iloc[i]
        rTop_val = ph.iloc[i] if not pd.isna(ph.iloc[i]) else high.iloc[i]
        rBot_val = r_yLoc[i] if not pd.isna(r_yLoc[i]) else high.iloc[i]
        
        if pd.isna(sTop_val) or pd.isna(sBot_val) or pd.isna(rTop_val) or pd.isna(rBot_val):
            continue
        
        # Breakout detection
        sBreak_prev = sBreak.iloc[i-1]
        rBreak_prev = rBreak.iloc[i-1]
        
        cu = close.iloc[i] < sBot_val and close.iloc[i-1] >= sBot_val
        co = close.iloc[i] > rTop_val and close.iloc[i-1] <= rTop_val
        
        if cu and not sBreak_prev:
            sBreak.iloc[i] = True
        else:
            sBreak.iloc[i] = sBreak_prev
            
        if co and not rBreak_prev:
            rBreak.iloc[i] = True
        else:
            rBreak.iloc[i] = rBreak_prev
        
        # Retest conditions for support
        bars_since_break = 0
        if sBreak.iloc[i]:
            for j in range(i-1, -1, -1):
                if not sBreak.iloc[j]:
                    bars_since_break = i - j
                    break
        
        s1 = bars_since_break > input_retSince and high.iloc[i] >= sTop_val and close.iloc[i] <= sBot_val
        s2 = bars_since_break > input_retSince and high.iloc[i] >= sTop_val and close.iloc[i] >= sBot_val and close.iloc[i] <= sTop_val
        s3 = bars_since_break > input_retSince and high.iloc[i] >= sBot_val and high.iloc[i] <= sTop_val
        s4 = bars_since_break > input_retSince and high.iloc[i] >= sBot_val and high.iloc[i] <= sTop_val and close.iloc[i] < sBot_val
        
        sRetActive = s1 or s2 or s3 or s4
        
        # Retest conditions for resistance
        bars_since_break_r = 0
        if rBreak.iloc[i]:
            for j in range(i-1, -1, -1):
                if not rBreak.iloc[j]:
                    bars_since_break_r = i - j
                    break
        
        r1 = bars_since_break_r > input_retSince and low.iloc[i] <= rBot_val and close.iloc[i] >= rTop_val
        r2 = bars_since_break_r > input_retSince and low.iloc[i] <= rBot_val and close.iloc[i] <= rTop_val and close.iloc[i] >= rBot_val
        r3 = bars_since_break_r > input_retSince and low.iloc[i] <= rTop_val and low.iloc[i] >= rBot_val
        r4 = bars_since_break_r > input_retSince and low.iloc[i] <= rTop_val and low.iloc[i] >= rBot_val and close.iloc[i] > rTop_val
        
        rRetActive = r1 or r2 or r3 or r4
        
        # Check for new retest event
        retEvent_s = sRetActive and (i == 0 or not sRetActive)
        retEvent_r = rRetActive and (i == 0 or not rRetActive)
        
        if retEvent_s:
            retOccurred_s.iloc[i] = True
        elif i > 0 and retOccurred_s.iloc[i-1]:
            retOccurred_s.iloc[i] = False
        
        if retEvent_r:
            retOccurred_r.iloc[i] = True
        elif i > 0 and retOccurred_r.iloc[i-1]:
            retOccurred_r.iloc[i] = False
        
        if retEvent_s and not retOccurred_s.iloc[i-1] if i > 0 else True:
            bars_since_ret = 0
            for j in range(i-1, -1, -1):
                if sRetActive and (j == 0 or not sRetActive):
                    bars_since_ret = i - j
                    break
                if not sRetActive:
                    bars_since_ret = i - j
                    break
            
            if bars_since_ret > 0 and bars_since_ret <= input_retValid:
                sRetValid.iloc[i] = True
                retOccurred_s.iloc[i] = True
        
        if retEvent_r and not retOccurred_r.iloc[i-1] if i > 0 else True:
            bars_since_ret_r = 0
            for j in range(i-1, -1, -1):
                if rRetActive and (j == 0 or not rRetActive):
                    bars_since_ret_r = i - j
                    break
                if not rRetActive:
                    bars_since_ret_r = i - j
                    break
            
            if bars_since_ret_r > 0 and bars_since_ret_r <= input_retValid:
                rRetValid.iloc[i] = True
                retOccurred_r.iloc[i] = True
    
    # Generate entries based on conditions
    entries = []
    trade_num = 1
    
    long_enabled = tradeDirection in ["Long", "Both"]
    short_enabled = tradeDirection in ["Short", "Both"]
    
    for i in range(len(df)):
        if i < bb + 2:
            continue
            
        if input_retest:
            if long_enabled and sRetValid.iloc[i]:
                ts = int(df['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
                trade_num += 1
            
            if short_enabled and rRetValid.iloc[i]:
                ts = int(df['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
                trade_num += 1
    
    return entries