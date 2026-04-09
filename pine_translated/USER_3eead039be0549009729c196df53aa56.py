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
    
    # Parameters (from Pine Script inputs)
    atrLength = 14
    atrMultiplier = 1.5
    takeProfitRatio = 1.5
    tradeDirection = 'Both'
    ssLength = 20
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    # Repainting mode: 'On', 'Off: Candle Confirmation', 'Off: High & Low'
    # Using 'Off: Candle Confirmation' for non-repainting entries
    rTon = False
    rTcc = True
    rThv = False
    
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=length).mean()
        return atr
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Pivot calculations
    bb = input_lookback
    
    pl = pd.Series(index=df.index, dtype=float)
    ph = pd.Series(index=df.index, dtype=float)
    
    for i in range(bb, len(df)):
        window_low = low.iloc[i-bb:i+1]
        window_high = high.iloc[i-bb:i+1]
        pl.iloc[i] = window_low.min()
        ph.iloc[i] = window_high.max()
    
    # Box boundaries
    s_yLoc = pd.Series(index=df.index, dtype=float)
    r_yLoc = pd.Series(index=df.index, dtype=float)
    sTop = pd.Series(index=df.index, dtype=float)
    sBot = pd.Series(index=df.index, dtype=float)
    rTop = pd.Series(index=df.index, dtype=float)
    rBot = pd.Series(index=df.index, dtype=float)
    
    for i in range(bb, len(df)):
        s_yLoc.iloc[i] = low.iloc[bb + 1] if low.iloc[bb + 1] > low.iloc[bb - 1] else low.iloc[bb - 1]
        r_yLoc.iloc[i] = high.iloc[bb + 1] if high.iloc[bb + 1] > high.iloc[bb - 1] else high.iloc[bb - 1]
        sTop.iloc[i] = pl.iloc[i]
        sBot.iloc[i] = s_yLoc.iloc[i]
        rTop.iloc[i] = ph.iloc[i]
        rBot.iloc[i] = r_yLoc.iloc[i]
    
    # ATR
    atr = wilder_atr(high, low, close, atrLength)
    
    # Super Smoother Filter
    src = close
    a1 = np.exp(-np.sqrt(2) * np.pi / ssLength)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / ssLength)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    
    SuperSmoother = pd.Series(index=df.index, dtype=float)
    for i in range(2, len(df)):
        if pd.isna(SuperSmoother.iloc[i-1]):
            SuperSmoother.iloc[i] = c1 * (src.iloc[i] + src.iloc[i-1].fillna(src.iloc[i])) / 2 + c2 * src.iloc[i-1].fillna(src.iloc[i]) + c3 * src.iloc[i-2].fillna(src.iloc[i-1])
        else:
            SuperSmoother.iloc[i] = c1 * (src.iloc[i] + src.iloc[i-1]) / 2 + c2 * SuperSmoother.iloc[i-1] + c3 * SuperSmoother.iloc[i-2]
    
    # Box values at breakout points
    sBot_at_break = pd.Series(index=df.index, dtype=float)
    rTop_at_break = pd.Series(index=df.index, dtype=float)
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)
    
    for i in range(bb + 1, len(df)):
        sBreak_idx = i - 1
        rBreak_idx = i - 1
        if sBreak_idx >= bb:
            sBot_at_break.iloc[i] = sBot.iloc[sBreak_idx]
        if rBreak_idx >= bb:
            rTop_at_break.iloc[i] = rTop.iloc[rBreak_idx]
    
    # Crossover and Crossunder detection
    cu = (close < sBot) & (close.shift(1) >= sBot)
    co = (close > rTop) & (close.shift(1) <= rTop)
    
    # Breakout detection with state
    for i in range(bb, len(df)):
        if cu.iloc[i] and pd.isna(sBreak.iloc[i-1]):
            sBreak.iloc[i] = True
            sBreak_idx = i
            if sBreak_idx >= bb:
                sBot_at_break.iloc[i] = sBot.iloc[sBreak_idx]
        else:
            sBreak.iloc[i] = sBreak.iloc[i-1]
        
        if co.iloc[i] and pd.isna(rBreak.iloc[i-1]):
            rBreak.iloc[i] = True
            rBreak_idx = i
            if rBreak_idx >= bb:
                rTop_at_break.iloc[i] = rTop.iloc[rBreak_idx]
        else:
            rBreak.iloc[i] = rBreak.iloc[i-1]
        
        # Reset on pivot change
        if i > 0:
            if not pd.isna(pl.iloc[i]) and pl.iloc[i] != pl.iloc[i-1]:
                if pd.isna(sBreak.iloc[i-1]):
                    sBreak.iloc[i] = False
            if not pd.isna(ph.iloc[i]) and ph.iloc[i] != ph.iloc[i-1]:
                if pd.isna(rBreak.iloc[i-1]):
                    rBreak.iloc[i] = False
    
    # Retest conditions for resistance (long)
    bars_since_rBreak = pd.Series(index=df.index, dtype=int)
    for i in range(len(df)):
        if rBreak.iloc[i]:
            bars_since_rBreak.iloc[i] = 0
        elif i > 0:
            bars_since_rBreak.iloc[i] = bars_since_rBreak.iloc[i-1] + 1 if pd.notna(bars_since_rBreak.iloc[i-1]) else 0
    
    r1 = (bars_since_rBreak > input_retSince) & (low <= rBot) & (close >= rTop)
    r2 = (bars_since_rBreak > input_retSince) & (low <= rBot) & (close <= rTop) & (close >= rBot)
    r3 = (bars_since_rBreak > input_retSince) & (low <= rTop) & (low >= rBot)
    r4 = (bars_since_rBreak > input_retSince) & (low <= rTop) & (low >= rBot) & (close > rTop)
    
    rRetActive = r1 | r2 | r3 | r4
    
    # Retest conditions for support (short)
    bars_since_sBreak = pd.Series(index=df.index, dtype=int)
    for i in range(len(df)):
        if sBreak.iloc[i]:
            bars_since_sBreak.iloc[i] = 0
        elif i > 0:
            bars_since_sBreak.iloc[i] = bars_since_sBreak.iloc[i-1] + 1 if pd.notna(bars_since_sBreak.iloc[i-1]) else 0
    
    s1 = (bars_since_sBreak > input_retSince) & (high >= sTop) & (close <= sBot)
    s2 = (bars_since_sBreak > input_retSince) & (high >= sTop) & (close >= sBot) & (close <= sTop)
    s3 = (bars_since_sBreak > input_retSince) & (high >= sBot) & (high <= sTop)
    s4 = (bars_since_sBreak > input_retSince) & (high >= sBot) & (high <= sTop) & (close < sBot)
    
    sRetActive = s1 | s2 | s3 | s4
    
    # Retest validity (using non-repainting mode: rTcc)
    rRetEvent = rRetActive & ~rRetActive.shift(1).fillna(False)
    sRetEvent = sRetActive & ~sRetActive.shift(1).fillna(False)
    
    # Calculate bars since retest event
    bars_since_rRetEvent = pd.Series(index=df.index, dtype=int)
    bars_since_sRetEvent = pd.Series(index=df.index, dtype=int)
    
    for i in range(len(df)):
        if rRetEvent.iloc[i]:
            bars_since_rRetEvent.iloc[i] = 0
        elif i > 0 and bars_since_rRetEvent.iloc[i-1] >= 0:
            bars_since_rRetEvent.iloc[i] = bars_since_rRetEvent.iloc[i-1] + 1
        else:
            bars_since_rRetEvent.iloc[i] = 0
    
    for i in range(len(df)):
        if sRetEvent.iloc[i]:
            bars_since_sRetEvent.iloc[i] = 0
        elif i > 0 and bars_since_sRetEvent.iloc[i-1] >= 0:
            bars_since_sRetEvent.iloc[i] = bars_since_sRetEvent.iloc[i-1] + 1
        else:
            bars_since_sRetEvent.iloc[i] = 0
    
    # Retest conditions based on mode
    if rTon:
        rRetConditions = close >= rTop_at_break
    elif rThv:
        rRetConditions = high >= rTop_at_break
    else:
        rRetConditions = close >= rTop_at_break
    
    if rTon:
        sRetConditions = close <= sBot_at_break
    elif rThv:
        sRetConditions = low <= sBot_at_break
    else:
        sRetConditions = close <= sBot_at_break
    
    rRetValid = (bars_since_rRetEvent > 0) & (bars_since_rRetEvent <= input_retValid) & rRetConditions
    sRetValid = (bars_since_sRetEvent > 0) & (bars_since_sRetEvent <= input_retValid) & sRetConditions
    
    # Entry conditions
    long_condition = rBreak & rRetValid
    short_condition = sBreak & sRetValid
    
    # Generate entries
    entries = []
    trade_num = 1
    
    start_idx = max(ssLength, bb, input_retSince + input_retValid + 1)
    
    for i in range(start_idx, len(df)):
        if pd.isna(close.iloc[i]):
            continue
        
        entry_price = close.iloc[i]
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        # Long entry
        if long_condition.iloc[i] and (tradeDirection in ['Long', 'Both']):
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
        
        # Short entry
        if short_condition.iloc[i] and (tradeDirection in ['Short', 'Both']):
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
    
    return entries