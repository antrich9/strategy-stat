import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.
    Returns list of dicts with entry signals.
    """
    # Strategy parameters (extracted from Pine Script inputs)
    lookback = 20
    atrLength = 14
    atrMultiplier = 1.5
    takeProfitRatio = 1.5
    silencePeriod = 10
    retSince = 2
    retValid = 2
    tradeDirection = "Both"
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate Wilder ATR manually
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atrLength, adjust=False).mean()
    
    # Calculate pivot points using Pine Script logic: ta.pivotlow(high, lb, lb)
    # A pivot low at index i is the lowest in window [i-lb, i+lb]
    bb = lookback
    
    pivot_low_vals = pd.Series(np.nan, index=df.index)
    pivot_high_vals = pd.Series(np.nan, index=df.index)
    
    for i in range(bb, len(df) - bb):
        window_low = low.iloc[i-bb:i+bb+1]
        window_high = high.iloc[i-bb:i+bb+1]
        if low.iloc[i] == window_low.min():
            pivot_low_vals.iloc[i] = low.iloc[i]
        if high.iloc[i] == window_high.max():
            pivot_high_vals.iloc[i] = high.iloc[i]
    
    # fixnan equivalent: forward fill then backward fill
    pivot_low_vals = pivot_low_vals.ffill().bfill()
    pivot_high_vals = pivot_high_vals.ffill().bfill()
    
    # Calculate s_yLoc and r_yLoc (box reference points from Pine Script)
    s_yLoc = pd.Series(index=df.index, dtype=float)
    r_yLoc = pd.Series(index=df.index, dtype=float)
    
    for i in range(bb + 1, len(df) - bb + 1):
        s_yLoc.iloc[i] = low.iloc[bb - 1] if low.iloc[bb + 1] > low.iloc[bb - 1] else low.iloc[bb + 1]
        r_yLoc.iloc[i] = high.iloc[bb + 1] if high.iloc[bb + 1] > high.iloc[bb - 1] else high.iloc[bb - 1]
    
    # Box levels: sBot = min(pl, s_yLoc), rTop = max(ph, r_yLoc)
    # For simplicity, use pivot values as the key levels
    sBot = pivot_low_vals.ffill()
    rTop = pivot_high_vals.ffill()
    
    # Breakout conditions: ta.crossover and ta.crossunder
    co = (close > rTop) & (close.shift(1) <= rTop.shift(1))
    cu = (close < sBot) & (close.shift(1) >= sBot.shift(1))
    
    # Track breakout state (var bool sBreak/rBreak in Pine)
    rBreak_active = pd.Series(False, index=df.index)
    sBreak_active = pd.Series(False, index=df.index)
    
    for i in range(1, len(df)):
        if co.iloc[i] and not rBreak_active.iloc[i-1]:
            rBreak_active.iloc[i] = True
        elif rBreak_active.iloc[i-1]:
            rBreak_active.iloc[i] = True
        
        if cu.iloc[i] and not sBreak_active.iloc[i-1]:
            sBreak_active.iloc[i] = True
        elif sBreak_active.iloc[i-1]:
            sBreak_active.iloc[i] = True
    
    # Retest conditions: price returns to broken level
    # Simplified: close crosses back to the level after breakout
    rRetest = rBreak_active & (close.shift(1) >= rTop.shift(1)) & (close < rTop)
    sRetest = sBreak_active & (close.shift(1) <= sBot.shift(1)) & (close > sBot)
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if np.isnan(atr.iloc[i]) or np.isnan(rTop.iloc[i]) or np.isnan(sBot.iloc[i]):
            continue
        
        direction = None
        
        # Long entry: resistance breakout (co) or resistance retest
        if tradeDirection in ["Long", "Both"]:
            if co.iloc[i]:
                direction = "long"
            elif rRetest.iloc[i]:
                direction = "long"
        
        # Short entry: support breakout (cu) or support retest
        if tradeDirection in ["Short", "Both"]:
            if cu.iloc[i]:
                direction = "short"
            elif sRetest.iloc[i]:
                direction = "short"
        
        if direction:
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries