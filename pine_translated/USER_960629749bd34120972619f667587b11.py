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
    
    # Calculate candle structure
    body = (df['close'] - df['open']).abs()
    candle_range = df['high'] - df['low']
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    total_wick = upper_wick + lower_wick
    
    # Prevent division by zero
    body_safe = body.replace(0, np.nan)
    candle_range_safe = candle_range.replace(0, np.nan)
    
    # FVG Filter parameters (inactive by default in original)
    inp1 = False  # Volume Filter
    inp2 = False  # ATR Filter
    inp3 = False  # Trend Filter
    
    # Volume filter
    volfilt = (df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5) if inp1 else pd.Series(True, index=df.index)
    
    # ATR filter (using Wilder ATR)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atr2 = atr / 1.5
    atrfilt = ((df['low'] - df['high'].shift(2) > atr2) | (df['low'].shift(2) - df['high'] > atr2)) if inp2 else pd.Series(True, index=df.index)
    
    # Trend filter
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2 if inp3 else pd.Series(True, index=df.index)
    locfilts = ~loc2 if inp3 else pd.Series(True, index=df.index)
    
    # Bullish FVG: low > high[2]
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    
    # Bearish FVG: high < low[2]
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    # Sharp turn detection (4H context - simplified to chart timeframe)
    # lastFVG tracks previous FVG direction
    last_fvg = 0  # 0 = none, 1 = bullish, -1 = bearish
    entries = []
    trade_num = 1
    
    for i in range(2, len(df)):
        # Skip if any indicator is NaN
        if pd.isna(bfvg.iloc[i]) or pd.isna(sfvg.iloc[i]):
            continue
        if pd.isna(locfiltb.iloc[i]) or pd.isna(locfilts.iloc[i]):
            continue
            
        current_bfvg = bfvg.iloc[i]
        current_sfvg = sfvg.iloc[i]
        
        # Sharp Turn Long Entry: Bullish FVG after Bearish FVG
        if current_bfvg and last_fvg == -1:
            entry_price = df['close'].iloc[i]
            ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
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
            last_fvg = 1
            
        # Sharp Turn Short Entry: Bearish FVG after Bullish FVG
        elif current_sfvg and last_fvg == 1:
            entry_price = df['close'].iloc[i]
            ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
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
            last_fvg = -1
            
        # Update lastFVG even if no entry
        elif current_bfvg:
            last_fvg = 1
        elif current_sfvg:
            last_fvg = -1
    
    return entries