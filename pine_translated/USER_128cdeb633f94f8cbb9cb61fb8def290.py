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
    atrLength = 14
    
    # Calculate Wilder ATR manually
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atrLength, adjust=False).mean()
    
    # Pivot calculation
    PP = 5
    pivots_high = pd.Series(np.nan, index=df.index)
    pivots_low = pd.Series(np.nan, index=df.index)
    
    for i in range(PP, len(df) - PP):
        is_high = True
        for j in range(1, PP + 1):
            if df['high'].iloc[i] <= df['high'].iloc[i + j] or df['high'].iloc[i] < df['high'].iloc[i - j]:
                is_high = False
                break
        if is_high:
            pivots_high.iloc[i] = df['high'].iloc[i]
        
        is_low = True
        for j in range(1, PP + 1):
            if df['low'].iloc[i] >= df['low'].iloc[i + j] or df['low'].iloc[i] > df['low'].iloc[i - j]:
                is_low = False
                break
        if is_low:
            pivots_low.iloc[i] = df['low'].iloc[i]
    
    # Store pivot data
    pivot_types = []
    pivot_values = []
    pivot_indices = []
    
    for i in range(len(df)):
        if not np.isnan(pivots_high.iloc[i]):
            pivot_types.append('H')
            pivot_values.append(pivots_high.iloc[i])
            pivot_indices.append(i)
        if not np.isnan(pivots_low.iloc[i]):
            pivot_types.append('L')
            pivot_values.append(pivots_low.iloc[i])
            pivot_indices.append(i)
    
    # Determine Major/Minor structure
    Bullish_Major_BoS = False
    Bearish_Major_BoS = False
    Bullish_Major_ChoCh = False
    Bearish_Major_ChoCh = False
    Bullish_Minor_BoS = False
    Bearish_Minor_BoS = False
    Bullish_Minor_ChoCh = False
    Bearish_Minor_ChoCh = False
    
    entries = []
    trade_num = 1
    
    # Track major structure levels
    prev_major_high_val = np.nan
    prev_major_low_val = np.nan
    prev_minor_high_val = np.nan
    prev_minor_low_val = np.nan
    
    major_high_idx = []
    major_low_idx = []
    major_high_val = []
    major_low_val = []
    
    minor_high_idx = []
    minor_low_idx = []
    minor_high_val = []
    minor_low_val = []
    
    for i in range(len(df)):
        # Update major levels on pivot detection
        if not np.isnan(pivots_high.iloc[i]):
            # Major high: highest recent high or significant high
            if len(major_high_val) == 0 or pivots_high.iloc[i] > major_high_val[-1]:
                major_high_idx.append(i)
                major_high_val.append(pivots_high.iloc[i])
                if len(major_high_val) > 2:
                    major_high_idx.pop(0)
                    major_high_val.pop(0)
            elif len(major_high_val) > 0:
                minor_high_idx.append(i)
                minor_high_val.append(pivots_high.iloc[i])
                if len(minor_high_val) > 2:
                    minor_high_idx.pop(0)
                    minor_high_val.pop(0)
        
        if not np.isnan(pivots_low.iloc[i]):
            if len(major_low_val) == 0 or pivots_low.iloc[i] < major_low_val[-1]:
                major_low_idx.append(i)
                major_low_val.append(pivots_low.iloc[i])
                if len(major_low_val) > 2:
                    major_low_idx.pop(0)
                    major_low_val.pop(0)
            elif len(major_low_val) > 0:
                minor_low_idx.append(i)
                minor_low_val.append(pivots_low.iloc[i])
                if len(minor_low_val) > 2:
                    minor_low_idx.pop(0)
                    minor_low_val.pop(0)
        
        # Detect BoS and ChoCh conditions
        if len(major_high_val) >= 2 and len(major_low_val) >= 2:
            current_high = major_high_val[-1]
            prev_high = major_high_val[-2]
            current_low = major_low_val[-1]
            prev_low = major_low_val[-2]
            
            if current_high > prev_high and not Bearish_Major_BoS:
                Bullish_Major_BoS = True
                Bearish_Major_BoS = False
            elif current_low < prev_low and not Bullish_Major_BoS:
                Bearish_Major_BoS = True
                Bullish_Major_BoS = False
            
            if current_high > prev_low and not Bearish_Major_ChoCh:
                Bullish_Major_ChoCh = True
                Bearish_Major_ChoCh = False
            elif current_low < prev_high and not Bullish_Major_ChoCh:
                Bearish_Major_ChoCh = True
                Bullish_Major_ChoCh = False
        
        if len(minor_high_val) >= 2 and len(minor_low_val) >= 2:
            current_minor_high = minor_high_val[-1]
            prev_minor_high = minor_high_val[-2]
            current_minor_low = minor_low_val[-1]
            prev_minor_low = minor_low_val[-2]
            
            if current_minor_high > prev_minor_high and not Bearish_Minor_BoS:
                Bullish_Minor_BoS = True
                Bearish_Minor_BoS = False
            elif current_minor_low < prev_minor_low and not Bullish_Minor_BoS:
                Bearish_Minor_BoS = True
                Bullish_Minor_BoS = False
            
            if current_minor_high > prev_minor_low and not Bearish_Minor_ChoCh:
                Bullish_Minor_ChoCh = True
                Bearish_Minor_ChoCh = False
            elif current_minor_low < prev_minor_high and not Bullish_Minor_ChoCh:
                Bearish_Minor_ChoCh = True
                Bearish_Minor_ChoCh = False
        
        # Entry logic based on structure detection
        if Bullish_Major_BoS or Bullish_Major_ChoCh or Bullish_Minor_BoS or Bullish_Minor_ChoCh:
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            Bullish_Major_BoS = False
            Bullish_Major_ChoCh = False
            Bullish_Minor_BoS = False
            Bullish_Minor_ChoCh = False
        
        if Bearish_Major_BoS or Bearish_Major_ChoCh or Bearish_Minor_BoS or Bearish_Minor_ChoCh:
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            Bearish_Major_BoS = False
            Bearish_Major_ChoCh = False
            Bearish_Minor_BoS = False
            Bearish_Minor_ChoCh = False
    
    return entries