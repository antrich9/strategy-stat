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
    entries = []
    trade_num = 1
    entered_fvgs = set()
    
    # Detect FVG zones using Pine Script logic from find_box()
    # Bullish FVG: low >= high[2], Bearish FVG: low[2] >= high
    x = pd.Series(0, index=df.index)
    x.iloc[2:] = np.where(
        df['low'].iloc[2:].values >= df['high'].iloc[:-2].values,
        1,
        np.where(
            df['low'].iloc[:-2].values >= df['high'].iloc[2:].values,
            -1,
            0
        )
    )
    
    # Calculate FVG top and bottom
    fvg_top = pd.Series(np.nan, index=df.index)
    fvg_bottom = pd.Series(np.nan, index=df.index)
    
    bull_mask = x == 1
    bear_mask = x == -1
    
    fvg_top.loc[bull_mask] = df['low'].loc[bull_mask]
    fvg_bottom.loc[bull_mask] = df['high'].shift(2).loc[bull_mask]
    fvg_top.loc[bear_mask] = df['low'].shift(2).loc[bear_mask]
    fvg_bottom.loc[bear_mask] = df['high'].loc[bear_mask]
    
    # Iterate through bars and check for entries
    for i in range(len(df)):
        if i == 0:
            continue
        
        # Bullish FVG entry: price enters from above (low < fvg_top)
        bull_entry = (
            x.iloc[i] == 1 and
            df['low'].iloc[i] < fvg_top.iloc[i] and
            df['close'].iloc[i-1] > fvg_top.iloc[i] and
            i not in entered_fvgs and
            pd.notna(fvg_top.iloc[i])
        )
        
        if bull_entry:
            entered_fvgs.add(i)
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i]/1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            continue
        
        # Bearish FVG entry: price enters from below (high > fvg_bottom)
        bear_entry = (
            x.iloc[i] == -1 and
            df['high'].iloc[i] > fvg_bottom.iloc[i] and
            df['close'].iloc[i-1] < fvg_bottom.iloc[i] and
            i not in entered_fvgs and
            pd.notna(fvg_bottom.iloc[i])
        )
        
        if bear_entry:
            entered_fvgs.add(i)
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i]/1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries