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
    PP = 5
    atrLength = 14
    
    # Calculate ATR using Wilder's method
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = pd.Series(index=tr.index, dtype=float)
    if len(tr) > 0:
        atr.iloc[atrLength - 1] = tr.iloc[:atrLength].mean()
        for i in range(atrLength, len(tr)):
            atr.iloc[i] = (atr.iloc[i - 1] * (atrLength - 1) + tr.iloc[i]) / atrLength
    
    # Pivot detection
    pivothigh = pd.Series(False, index=df.index)
    pivotlow = pd.Series(False, index=df.index)
    
    for i in range(PP, len(df) - PP):
        ph = df['high'].iloc[i]
        pl = df['low'].iloc[i]
        is_pivot_high = True
        is_pivot_low = True
        
        for j in range(i - PP, i + PP + 1):
            if j == i:
                continue
            if df['high'].iloc[j] >= ph:
                is_pivot_high = False
            if df['low'].iloc[j] <= pl:
                is_pivot_low = False
        
        if is_pivot_high:
            pivothigh.iloc[i] = True
        if is_pivot_low:
            pivotlow.iloc[i] = True
    
    # Zigzag and structure tracking
    zigzag_types = []
    zigzag_values = []
    zigzag_indices = []
    
    Major_HighLevel = np.nan
    Major_LowLevel = np.nan
    Major_HighIndex = 0
    Major_LowIndex = 0
    
    Bullish_Major_BoS = False
    Bearish_Major_BoS = False
    
    dtTradeTriggered = False
    dbTradeTriggered = False
    isLongOpen = False
    isShortOpen = False
    
    entries = []
    trade_num = 1
    
    last_high_idx = -1
    last_low_idx = -1
    
    for i in range(len(df)):
        ts = df['time'].iloc[i]
        dt_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        close_i = df['close'].iloc[i]
        
        if i < PP * 2 or pd.isna(atr.iloc[i]):
            continue
        
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        
        # Update Major levels
        if pivothigh.iloc[i]:
            if not np.isnan(Major_HighLevel):
                if current_high > Major_HighLevel:
                    Major_HighLevel = current_high
                    Major_HighIndex = i
            else:
                Major_HighLevel = current_high
                Major_HighIndex = i
            last_high_idx = i
        
        if pivotlow.iloc[i]:
            if not np.isnan(Major_LowLevel):
                if current_low < Major_LowLevel:
                    Major_LowLevel = current_low
                    Major_LowIndex = i
            else:
                Major_LowLevel = current_low
                Major_LowIndex = i
            last_low_idx = i
        
        # Structure detection
        Bullish_Major_BoS = False
        Bearish_Major_BoS = False
        
        # Check for Major Bullish BoS: Higher High after HH-LH-LL sequence
        if last_high_idx > 0 and last_low_idx > 0:
            high_idxs = [j for j in range(i) if pivothigh.iloc[j]]
            low_idxs = [j for j in range(i) if pivotlow.iloc[j]]
            
            if len(high_idxs) >= 2 and len(low_idxs) >= 2:
                h1 = df['high'].iloc[high_idxs[-2]]
                h2 = df['high'].iloc[high_idxs[-1]]
                l1 = df['low'].iloc[low_idxs[-2]]
                l2 = df['low'].iloc[low_idxs[-1]]
                
                # Bullish: HH -> LH -> LL followed by new HH
                if l2 < l1 and h2 < h1 and current_high > h2:
                    Bullish_Major_BoS = True
                    dtTradeTriggered = True
                
                # Bearish: LL -> HL -> HH followed by new LL
                if h2 > h1 and l2 < l1 and current_low < l2:
                    Bearish_Major_BoS = True
                    dbTradeTriggered = True
        
        # Entry execution
        if dtTradeTriggered and not isShortOpen:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': dt_str,
                'entry_price_guess': close_i,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_i,
                'raw_price_b': close_i
            })
            trade_num += 1
            isLongOpen = True
            dtTradeTriggered = False
        
        if dbTradeTriggered and not isLongOpen:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': dt_str,
                'entry_price_guess': close_i,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_i,
                'raw_price_b': close_i
            })
            trade_num += 1
            isShortOpen = True
            dbTradeTriggered = False
    
    return entries