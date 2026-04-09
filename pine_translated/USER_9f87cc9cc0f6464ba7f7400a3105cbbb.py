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
    
    # Strategy parameters (from input)
    bb = 20  # lookback range for pivot detection
    input_retSince = 2  # bars since breakout for retest detection
    
    # Trade direction setting
    tradeDirection = "Both"  # Can be "Long", "Short", or "Both"
    
    entries = []
    trade_num = 1
    
    # Calculate pivot points
    pl = pd.Series(index=df.index, dtype=float)  # pivot low
    ph = pd.Series(index=df.index, dtype=float)  # pivot high
    
    for i in range(bb, len(df) - bb):
        # pivotlow(low, bb, bb) - lowest low over bb bars ending at i-bb
        window_start = i - bb - bb + 1
        window_end = i - bb + 1
        if window_start >= 0 and window_end <= len(df):
            pl.iloc[i] = df['low'].iloc[window_start:window_end].min()
            ph.iloc[i] = df['high'].iloc[window_start:window_end].max()
    
    pl = pl.ffill()
    ph = ph.ffill()
    
    # Box boundaries for each bar
    sBot = pd.Series(index=df.index, dtype=float)  # support box bottom
    rTop = pd.Series(index=df.index, dtype=float)  # resistance box top
    
    for i in range(bb, len(df)):
        # s_yLoc = low[bb + 1] > low[bb - 1] ? low[bb - 1] : low[bb + 1]
        if df['low'].iloc[i - bb - 1] > df['low'].iloc[i - bb + 1]:
            sBot.iloc[i] = df['low'].iloc[i - bb + 1]
        else:
            sBot.iloc[i] = df['low'].iloc[i - bb - 1]
        
        # r_yLoc = high[bb + 1] > high[bb - 1] ? high[bb + 1] : high[bb - 1]
        if df['high'].iloc[i - bb - 1] > df['high'].iloc[i - bb + 1]:
            rTop.iloc[i] = df['high'].iloc[i - bb - 1]
        else:
            rTop.iloc[i] = df['high'].iloc[i - bb + 1]
    
    sBot = sBot.ffill()
    rTop = rTop.ffill()
    
    # Detect breakouts
    cu = pd.Series(False, index=df.index)  # support breakout (crossunder)
    co = pd.Series(False, index=df.index)  # resistance breakout (crossover)
    
    for i in range(1, len(df)):
        if pd.notna(sBot.iloc[i]) and pd.notna(sBot.iloc[i - 1]):
            # crossunder(close, box.get_bottom(sBox)) - price breaks below support
            cu.iloc[i] = df['close'].iloc[i] < sBot.iloc[i] and df['close'].iloc[i - 1] >= sBot.iloc[i - 1]
        
        if pd.notna(rTop.iloc[i]) and pd.notna(rTop.iloc[i - 1]):
            # crossover(close, box.get_top(rBox)) - price breaks above resistance
            co.iloc[i] = df['close'].iloc[i] > rTop.iloc[i] and df['close'].iloc[i - 1] <= rTop.iloc[i - 1]
    
    # Track breakout states
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)
    
    for i in range(len(df)):
        # Support breakout
        if cu.iloc[i] and not sBreak.iloc[i - 1] if i > 0 else cu.iloc[i]:
            sBreak.iloc[i] = True
        elif i > 0:
            sBreak.iloc[i] = sBreak.iloc[i - 1]
        
        # Resistance breakout
        if co.iloc[i] and not rBreak.iloc[i - 1] if i > 0 else co.iloc[i]:
            rBreak.iloc[i] = True
        elif i > 0:
            rBreak.iloc[i] = rBreak.iloc[i - 1]
        
        # Reset on pivot change
        if pd.notna(pl.iloc[i]) and pd.notna(pl.iloc[i - 1]) and pl.iloc[i] != pl.iloc[i - 1]:
            sBreak.iloc[i] = False
        if pd.notna(ph.iloc[i]) and pd.notna(ph.iloc[i - 1]) and ph.iloc[i] != ph.iloc[i - 1]:
            rBreak.iloc[i] = False
    
    # Generate entries based on breakout conditions
    for i in range(len(df)):
        if pd.isna(df['close'].iloc[i]):
            continue
        
        # Long entry: resistance breakout (co) when rBreak is not already active
        long_condition = co.iloc[i] and not rBreak.iloc[i]
        
        # Short entry: support breakout (cu) when sBreak is not already active
        short_condition = cu.iloc[i] and not sBreak.iloc[i]
        
        # Check trade direction
        should_long = tradeDirection in ["Long", "Both"]
        should_short = tradeDirection in ["Short", "Both"]
        
        if long_condition and should_long:
            entry_price = df['close'].iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            
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
            rBreak.iloc[i] = True  # Mark breakout as triggered
        
        if short_condition and should_short:
            entry_price = df['close'].iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            
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
            sBreak.iloc[i] = True  # Mark breakout as triggered
    
    return entries