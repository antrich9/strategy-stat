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
    tradeDirection = 'Both'
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    input_repType = 'On'
    
    bb = input_lookback
    
    # Wilder ATR
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atrLength, adjust=False).mean()
    
    # Pivot Points
    pl = pd.Series(np.where(
        (low.shift(bb) <= low.rolling(bb*2+1).min()) &
        (low < low.shift(1)) &
        (low < low.shift(-1)) &
        (low.shift(bb) == low),
        low.shift(bb),
        np.nan
    ), index=low.index)
    pl = pl.where(pl.fffill().notna(), np.nan)
    pl = pl.where(pl.bfill().notna(), np.nan)
    
    ph = pd.Series(np.where(
        (high.shift(bb) >= high.rolling(bb*2+1).max()) &
        (high > high.shift(1)) &
        (high > high.shift(-1)) &
        (high.shift(bb) == high),
        high.shift(bb),
        np.nan
    ), index=high.index)
    ph = ph.where(ph.fffill().notna(), np.nan)
    ph = ph.where(ph.bfill().notna(), np.nan)
    
    # Box boundaries
    s_yLoc = pd.Series(np.where(low.shift(bb+1) > low.shift(bb-1), low.shift(bb-1), low.shift(bb+1)), index=low.index)
    r_yLoc = pd.Series(np.where(high.shift(bb+1) > high.shift(bb-1), high.shift(bb+1), high.shift(bb-1)), index=high.index)
    
    # Support box top/bot
    sTop = pd.Series(np.nan, index=low.index)
    sBot = pd.Series(np.nan, index=low.index)
    
    # Resistance box top/bot
    rTop = pd.Series(np.nan, index=high.index)
    rBot = pd.Series(np.nan, index=high.index)
    
    pl_change = pl.diff() != 0
    ph_change = ph.diff() != 0
    
    for i in range(bb + 1, len(df)):
        pl_idx = i - bb
        if pl_idx >= 0 and not pd.isna(pl.iloc[pl_idx]) and pl_change.iloc[i] if i < len(pl_change) else False:
            sTop.iloc[i] = low.iloc[pl_idx]
            sBot.iloc[i] = s_yLoc.iloc[pl_idx]
        elif not pd.isna(sTop.iloc[i-1]):
            sTop.iloc[i] = sTop.iloc[i-1]
            sBot.iloc[i] = sBot.iloc[i-1]
    
    for i in range(bb + 1, len(df)):
        ph_idx = i - bb
        if ph_idx >= 0 and not pd.isna(ph.iloc[ph_idx]) and ph_change.iloc[i] if i < len(ph_change) else False:
            rTop.iloc[i] = high.iloc[ph_idx]
            rBot.iloc[i] = r_yLoc.iloc[ph_idx]
        elif not pd.isna(rTop.iloc[i-1]):
            rTop.iloc[i] = rTop.iloc[i-1]
            rBot.iloc[i] = rBot.iloc[i-1]
    
    # Breakout flags
    rTon = input_repType == 'On'
    rTcc = input_repType == 'Off: Candle Confirmation'
    rThv = input_repType == 'Off: High & Low'
    
    sBreak = pd.Series(False, index=low.index)
    rBreak = pd.Series(False, index=high.index)
    
    prev_sBreak = False
    prev_rBreak = False
    
    for i in range(bb + 1, len(df)):
        if pd.isna(sBot.iloc[i]) or pd.isna(rTop.iloc[i]):
            prev_sBreak = False
            prev_rBreak = False
            continue
        
        cu_crossunder = close.iloc[i] < sBot.iloc[i] and close.iloc[i-1] >= sBot.iloc[i-1]
        co_crossover = close.iloc[i] > rTop.iloc[i] and close.iloc[i-1] <= rTop.iloc[i-1]
        
        if cu_crossunder and not prev_sBreak:
            sBreak.iloc[i] = True
            prev_sBreak = True
        if co_crossover and not prev_rBreak:
            rBreak.iloc[i] = True
            prev_rBreak = True
        
        if pl_change.iloc[i] if i < len(pl_change) else False:
            prev_sBreak = False
        if ph_change.iloc[i] if i < len(ph_change) else False:
            prev_rBreak = False
    
    # Retest conditions for support (long)
    s1 = sBreak & (high >= sTop) & (close <= sBot)
    s2 = sBreak & (high >= sTop) & (close >= sBot) & (close <= sTop)
    s3 = sBreak & (high >= sBot) & (high <= sTop)
    s4 = sBreak & (high >= sBot) & (high <= sTop) & (close < sBot)
    
    # Retest conditions for resistance (short)
    r1 = rBreak & (low <= rBot) & (close >= rTop)
    r2 = rBreak & (low <= rBot) & (close <= rTop) & (close >= rBot)
    r3 = rBreak & (low <= rTop) & (low >= rBot)
    r4 = rBreak & (low <= rTop) & (low >= rBot) & (close > rTop)
    
    # Calculate bars since
    bars_since_sBreak = np.zeros(len(df))
    bars_since_rBreak = np.zeros(len(df))
    
    for i in range(1, len(df)):
        if sBreak.iloc[i]:
            bars_since_sBreak[i] = 0
        else:
            bars_since_sBreak[i] = bars_since_sBreak[i-1] + 1 if not pd.isna(bars_since_sBreak[i-1]) else 0
        
        if rBreak.iloc[i]:
            bars_since_rBreak[i] = 0
        else:
            bars_since_rBreak[i] = bars_since_rBreak[i-1] + 1 if not pd.isna(bars_since_rBreak[i-1]) else 0
    
    bars_since_sBreak = pd.Series(bars_since_sBreak, index=df.index)
    bars_since_rBreak = pd.Series(bars_since_rBreak, index=df.index)
    
    # Retest valid check
    sRetActive = s1 | s2 | s3 | s4
    rRetActive = r1 | r2 | r3 | r4
    
    sRetValid = pd.Series(False, index=df.index)
    rRetValid = pd.Series(False, index=df.index)
    
    prev_sRetActive = False
    prev_rRetActive = False
    sRetOccurred = False
    rRetOccurred = False
    sRetEvent_bar = -1000
    rRetEvent_bar = -1000
    
    for i in range(bb + 1, len(df)):
        if sRetActive.iloc[i] and not prev_sRetActive:
            sRetEvent_bar = i
            sRetOccurred = False
        if rRetActive.iloc[i] and not prev_rRetActive:
            rRetEvent_bar = i
            rRetOccurred = False
        
        prev_sRetActive = sRetActive.iloc[i]
        prev_rRetActive = rRetActive.iloc[i]
        
        bars_since_sRetEvent = i - sRetEvent_bar if sRetEvent_bar >= 0 else 1000
        bars_since_rRetEvent = i - rRetEvent_bar if rRetEvent_bar >= 0 else 1000
        
        ret_val_s = sTop.iloc[sRetEvent_bar] if sRetEvent_bar >= 0 else np.nan
        ret_val_r = rBot.iloc[rRetEvent_bar] if rRetEvent_bar >= 0 else np.nan
        
        s_ret_conditions = (close.iloc[i] <= ret_val_s) if rTon else ((high.iloc[i] >= ret_val_s) if rThv else (close.iloc[i] <= ret_val_s and True))
        r_ret_conditions = (close.iloc[i] >= ret_val_r) if rTon else ((low.iloc[i] <= ret_val_r) if rThv else (close.iloc[i] >= ret_val_r and True))
        
        if bars_since_sRetEvent > input_retSince and bars_since_sRetEvent <= input_retValid and s_ret_conditions and not sRetOccurred:
            sRetValid.iloc[i] = True
            sRetOccurred = True
        
        if bars_since_rRetEvent > input_retSince and bars_since_rRetEvent <= input_retValid and r_ret_conditions and not rRetOccurred:
            rRetValid.iloc[i] = True
            rRetOccurred = True
        
        if bars_since_sRetEvent > input_retValid:
            sRetOccurred = False
        if bars_since_rRetEvent > input_retValid:
            rRetOccurred = False
        
        if pl_change.iloc[i] if i < len(pl_change) else False:
            sRetOccurred = False
        if ph_change.iloc[i] if i < len(ph_change) else False:
            rRetOccurred = False
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(bb + 1, len(df)):
        if pd.isna(sRetValid.iloc[i]) or pd.isna(rRetValid.iloc[i]):
            continue
        
        direction = None
        if sRetValid.iloc[i] and tradeDirection in ['Long', 'Both']:
            direction = 'long'
        elif rRetValid.iloc[i] and tradeDirection in ['Short', 'Both']:
            direction = 'short'
        
        if direction is not None:
            entry_price = close.iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
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