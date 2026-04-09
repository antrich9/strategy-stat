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
    
    # Strategy parameters (matching Pine Script inputs)
    atrLength = 14
    jmaLength = 14
    volatilityMultiplier = 2.0
    bb = 20  # lookback for pivot
    input_retSince = 2
    input_retValid = 2
    tradeDirection = "Both"
    
    # Calculate JMA (simplified as SMA in the original)
    jma = df['close'].rolling(jmaLength).mean()
    
    # Calculate volatility (ATR * multiplier)
    atr = df['high'] - df['low']
    atr = atr.rolling(atrLength).mean()  # Simplified ATR
    volatility = atr * volatilityMultiplier
    
    # Calculate bands
    upperBand = jma + volatility
    lowerBand = jma - volatility
    
    # Calculate pivot points (Support and Resistance)
    pl = df['low'].rolling(window=bb+1).min().shift(1)  # pivotlow approximation
    ph = df['high'].rolling(window=bb+1).max().shift(1)  # pivothigh approximation
    
    # Box values (using current bar values for simplicity)
    sBot = lowerBand  # support box bottom
    sTop = lowerBand   # support box top
    rBot = upperBand   # resistance box bottom
    rTop = upperBand   # resistance box top
    
    # Breakout detection
    # co: crossover(close, rTop) - breakout above resistance
    # cu: crossunder(close, sBot) - breakout below support
    co = pd.Series(False, index=df.index)
    cu = pd.Series(False, index=df.index)
    
    for i in range(1, len(df)):
        if pd.notna(rTop.iloc[i-1]) and pd.notna(df['close'].iloc[i-1]):
            if df['close'].iloc[i] > rTop.iloc[i] and df['close'].iloc[i-1] <= rTop.iloc[i-1]:
                co.iloc[i] = True
        if pd.notna(sBot.iloc[i-1]) and pd.notna(df['close'].iloc[i-1]):
            if df['close'].iloc[i] < sBot.iloc[i] and df['close'].iloc[i-1] >= sBot.iloc[i-1]:
                cu.iloc[i] = True
    
    # Breakout flags
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)
    
    for i in range(1, len(df)):
        if cu.iloc[i] and not pd.notna(sBreak.iloc[i-1]):
            sBreak.iloc[i] = True
        if co.iloc[i] and not pd.notna(rBreak.iloc[i-1]):
            rBreak.iloc[i] = True
        # Reset on pivot change
        if i > 0:
            if pd.notna(pl.iloc[i]) and pd.notna(pl.iloc[i-1]) and pl.iloc[i] != pl.iloc[i-1]:
                sBreak.iloc[i] = False
            if pd.notna(ph.iloc[i]) and pd.notna(ph.iloc[i-1]) and ph.iloc[i] != ph.iloc[i-1]:
                rBreak.iloc[i] = False
    
    # Bars since functions
    def barssince(cond_series, current_idx):
        count = 0
        for j in range(current_idx - 1, -1, -1):
            if pd.notna(cond_series.iloc[j]) and cond_series.iloc[j]:
                return count
            count += 1
        return count + 1
    
    # Retest conditions (support side - for long entries)
    sRetValid = pd.Series(False, index=df.index)
    for i in range(bb + input_retSince + 1, len(df)):
        if pd.notna(sBreak.iloc[i]) and sBreak.iloc[i]:
            bars_since_break = barssince(sBreak, i)
            if bars_since_break > input_retSince:
                high_val = df['high'].iloc[i]
                low_val = df['low'].iloc[i]
                close_val = df['close'].iloc[i]
                sTop_val = sTop.iloc[i] if pd.notna(sTop.iloc[i]) else 0
                sBot_val = sBot.iloc[i] if pd.notna(sBot.iloc[i]) else 0
                
                if bars_since_break <= input_retValid:
                    cond1 = high_val >= sTop_val and close_val <= sBot_val
                    cond2 = high_val >= sTop_val and close_val >= sBot_val and close_val <= sTop_val
                    cond3 = high_val >= sBot_val and high_val <= sTop_val
                    cond4 = high_val >= sBot_val and high_val <= sTop_val and close_val < sBot_val
                    if cond1 or cond2 or cond3 or cond4:
                        sRetValid.iloc[i] = True
    
    # Retest conditions (resistance side - for short entries)
    rRetValid = pd.Series(False, index=df.index)
    for i in range(bb + input_retSince + 1, len(df)):
        if pd.notna(rBreak.iloc[i]) and rBreak.iloc[i]:
            bars_since_break = barssince(rBreak, i)
            if bars_since_break > input_retSince:
                high_val = df['high'].iloc[i]
                low_val = df['low'].iloc[i]
                close_val = df['close'].iloc[i]
                rTop_val = rTop.iloc[i] if pd.notna(rTop.iloc[i]) else 0
                rBot_val = rBot.iloc[i] if pd.notna(rBot.iloc[i]) else 0
                
                if bars_since_break <= input_retValid:
                    cond1 = low_val <= rBot_val and close_val >= rTop_val
                    cond2 = low_val <= rBot_val and close_val <= rTop_val and close_val >= rBot_val
                    cond3 = low_val <= rTop_val and low_val >= rBot_val
                    cond4 = low_val <= rTop_val and low_val >= rBot_val and close_val > rTop_val
                    if cond1 or cond2 or cond3 or cond4:
                        rRetValid.iloc[i] = True
    
    # Determine trade direction filtering
    allow_long = tradeDirection in ["Long", "Both"]
    allow_short = tradeDirection in ["Short", "Both"]
    
    # Build entries list
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(df['close'].iloc[i]):
            continue
        
        entry_price = df['close'].iloc[i]
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts / 1000 if ts > 1e10 else ts, tz=timezone.utc).isoformat()
        
        # Long entries on support retest valid
        if allow_long and sRetValid.iloc[i]:
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
        
        # Short entries on resistance retest valid
        if allow_short and rRetValid.iloc[i]:
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