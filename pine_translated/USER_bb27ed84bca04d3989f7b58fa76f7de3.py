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
    
    # Parameters from Pine Script
    atrLength = 14
    atrMultiplier = 1.5
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    rTon = True
    rTcc = False
    rThv = False
    bb = input_lookback
    
    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']
    
    # Calculate ATR using Wilder's method
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/atrLength, adjust=False).mean()
    
    # Calculate pivot high and low
    pl = pd.Series(index=df.index, dtype=float)
    ph = pd.Series(index=df.index, dtype=float)
    
    for i in range(bb, len(df)):
        pl_vals = low.iloc[max(0, i-bb):i+1]
        ph_vals = high.iloc[max(0, i-bb):i+1]
        
        min_idx = pl_vals.idxmin()
        max_idx = ph_vals.idxmax()
        
        if min_idx == i:
            pl.iloc[i] = low.iloc[i]
        if max_idx == i:
            ph.iloc[i] = high.iloc[i]
    
    pl = pl.fillna(method='ffill')
    ph = ph.fillna(method='ffill')
    
    # Box locations
    s_yLoc = np.where(low.shift(bb + 1) > low.shift(bb - 1), low.shift(bb - 1), low.shift(bb + 1))
    r_yLoc = np.where(high.shift(bb + 1) > high.shift(bb - 1), high.shift(bb + 1), high.shift(bb - 1))
    
    sBot_vals = low.shift(bb - 1).where(low.shift(bb + 1) > low.shift(bb - 1), low.shift(bb + 1))
    sTop_vals = pl.where(low.shift(bb + 1) > low.shift(bb - 1), pl)
    
    rTop_vals = high.shift(bb - 1).where(high.shift(bb + 1) > high.shift(bb - 1), high.shift(bb + 1))
    rBot_vals = ph.where(high.shift(bb + 1) > high.shift(bb - 1), ph)
    
    # Change signals for boxes
    sBox_change = pl != pl.shift(1)
    rBox_change = ph != ph.shift(1)
    
    # Breakout conditions
    cu_base = (close < sBot_vals) & (close.shift(1) >= sBot_vals.shift(1))
    co_base = (close > rTop_vals) & (close.shift(1) <= rTop_vals.shift(1))
    
    # Repaint conditions
    cu = cu_base  # rTon case (default based on inputs)
    co = co_base
    
    # Calculate ATR-based levels
    stopLossLong = close - (atr * atrMultiplier)
    stopLossShort = close + (atr * atrMultiplier)
    takeProfitLong = close + ((atr * atrMultiplier) * 1.5)
    takeProfitShort = close - ((atr * atrMultiplier) * 1.5)
    
    # Track breakout states
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)
    
    # Support retest conditions
    s1 = sBreak & (high >= sTop_vals) & (close <= sBot_vals)
    s2 = sBreak & (high >= sTop_vals) & (close >= sBot_vals) & (close <= sTop_vals)
    s3 = sBreak & (high >= sBot_vals) & (high <= sTop_vals)
    s4 = sBreak & (high >= sBot_vals) & (high <= sTop_vals) & (close < sBot_vals)
    
    # Resistance retest conditions
    r1 = rBreak & (low <= rBot_vals) & (close >= rTop_vals)
    r2 = rBreak & (low <= rBot_vals) & (close <= rTop_vals) & (close >= rBot_vals)
    r3 = rBreak & (low <= rTop_vals) & (low >= rBot_vals)
    r4 = rBreak & (low <= rTop_vals) & (low >= rBot_vals) & (close > rTop_vals)
    
    # Active retest tracking
    sRetActive = s1 | s2 | s3 | s4
    rRetActive = r1 | r2 | r3 | r4
    
    # Track retest events
    sRetEvent = pd.Series(False, index=df.index)
    rRetEvent = pd.Series(False, index=df.index)
    sRetOccurred = pd.Series(False, index=df.index)
    rRetOccurred = pd.Series(False, index=df.index)
    sRetValue = pd.Series(np.nan, index=df.index, dtype=float)
    rRetValue = pd.Series(np.nan, index=df.index, dtype=float)
    sRetValid = pd.Series(False, index=df.index)
    rRetValid = pd.Series(False, index=df.index)
    
    # Store retest bar indices and prices
    sRetBars = {}
    rRetBars = {}
    
    for i in range(len(df)):
        # Update box boundaries at current bar
        if not pd.isna(pl.iloc[i]) and pl.iloc[i] != pl.iloc[i-1] if i > 0 else True:
            sBot = sBot_vals.iloc[i]
            sTop = sTop_vals.iloc[i]
        
        if not pd.isna(ph.iloc[i]) and ph.iloc[i] != ph.iloc[i-1] if i > 0 else True:
            rBot = rBot_vals.iloc[i]
            rTop = rTop_vals.iloc[i]
        
        # Handle support breakout
        if i > 0:
            if cu.iloc[i] and pd.isna(sBreak.iloc[i-1]):
                sBreak.iloc[i] = True
            if sBox_change.iloc[i] and pd.isna(sBreak.iloc[i]):
                sBreak.iloc[i] = np.nan
                if i + bb < len(df):
                    sBot = sBot_vals.iloc[i + bb]
                    sTop = pl.iloc[i + bb]
        
        # Handle resistance breakout
        if i > 0:
            if co.iloc[i] and pd.isna(rBreak.iloc[i-1]):
                rBreak.iloc[i] = True
            if rBox_change.iloc[i] and pd.isna(rBreak.iloc[i]):
                rBreak.iloc[i] = np.nan
                if i + bb < len(df):
                    rBot = rBot_vals.iloc[i + bb]
                    rTop = ph.iloc[i + bb]
        
        # Support retest event detection
        if sRetActive.iloc[i] and not sRetActive.iloc[i-1] if i > 0 else False:
            sRetEvent.iloc[i] = True
            sRetOccurred.iloc[i] = True
            sRetBars[i] = {'price': high.iloc[i], 'bot': sBot_vals.iloc[i] if not pd.isna(sBot_vals.iloc[i]) else sBot, 'top': sTop_vals.iloc[i] if not pd.isna(sTop_vals.iloc[i]) else sTop}
            sRetValue.iloc[i] = high.iloc[i]
        
        # Resistance retest event detection
        if rRetActive.iloc[i] and not rRetActive.iloc[i-1] if i > 0 else False:
            rRetEvent.iloc[i] = True
            rRetOccurred.iloc[i] = True
            rRetBars[i] = {'price': low.iloc[i], 'bot': rBot_vals.iloc[i] if not pd.isna(rBot_vals.iloc[i]) else rBot, 'top': rTop_vals.iloc[i] if not pd.isna(rTop_vals.iloc[i]) else rTop}
            rRetValue.iloc[i] = low.iloc[i]
        
        # Process support retest validation
        for bar_idx, ret_data in list(sRetBars.items()):
            bars_since = i - bar_idx
            if bars_since > 0 and bars_since <= input_retValid:
                ret_val = ret_data['price']
                if close.iloc[i] <= ret_val:
                    if not sRetOccurred.iloc[i]:
                        sRetValid.iloc[i] = True
                        sRetOccurred.iloc[i] = True
                        sRetValue.iloc[i] = ret_val
            if bars_since > input_retValid:
                del sRetBars[bar_idx]
        
        # Process resistance retest validation
        for bar_idx, ret_data in list(rRetBars.items()):
            bars_since = i - bar_idx
            if bars_since > 0 and bars_since <= input_retValid:
                ret_val = ret_data['price']
                if close.iloc[i] >= ret_val:
                    if not rRetOccurred.iloc[i]:
                        rRetValid.iloc[i] = True
                        rRetOccurred.iloc[i] = True
                        rRetValue.iloc[i] = ret_val
            if bars_since > input_retValid:
                del rRetBars[bar_idx]
        
        # Reset retest occurred flags when new event
        if sRetEvent.iloc[i]:
            sRetOccurred.iloc[i] = False
        if rRetEvent.iloc[i]:
            rRetOccurred.iloc[i] = False
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i < bb + input_retSince + input_retValid:
            continue
        
        if sRetValid.iloc[i] and not pd.isna(sBreak.iloc[i]):
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = close.iloc[i]
            
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
        
        if rRetValid.iloc[i] and not pd.isna(rBreak.iloc[i]):
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = close.iloc[i]
            
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
    
    return entries