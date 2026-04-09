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
    
    # Fair Value Gaps - Bullish detection
    bull_fvg_cond = (
        (df['low'] > df['high'].shift(2)) &
        (df['close'].shift(1) > df['high'].shift(2)) &
        (df['open'].shift(2) < df['close'].shift(2)) &
        (df['open'].shift(1) < df['close'].shift(1)) &
        (df['open'] < df['close'])
    )
    
    # Fair Value Gaps - Bearish detection
    bear_fvg_cond = (
        (df['high'] < df['low'].shift(2)) &
        (df['close'].shift(1) < df['low'].shift(2)) &
        (df['open'].shift(2) > df['close'].shift(2)) &
        (df['open'].shift(1) > df['close'].shift(1)) &
        (df['open'] > df['close'])
    )
    
    # Opening Gaps - Bullish detection
    bull_og_cond = df['low'] > df['high'].shift(1)
    
    # Opening Gaps - Bearish detection
    bear_og_cond = df['high'] < df['low'].shift(1)
    
    # Volume Imbalances - Bullish detection
    bull_gap_top = np.minimum(df['close'], df['open'])
    bull_gap_btm = np.maximum(df['close'].shift(1), df['open'].shift(1))
    bull_vi_cond = (
        (df['open'] > df['close'].shift(1)) &
        (df['high'].shift(1) > df['low']) &
        (df['close'] > df['close'].shift(1)) &
        (df['open'] > df['open'].shift(1)) &
        (df['high'].shift(1) < bull_gap_top)
    )
    
    # Volume Imbalances - Bearish detection
    bear_gap_top = np.minimum(df['close'].shift(1), df['open'].shift(1))
    bear_gap_btm = np.maximum(df['close'], df['open'])
    bear_vi_cond = (
        (df['open'] < df['close'].shift(1)) &
        (df['low'].shift(1) < df['high']) &
        (df['close'] < df['close'].shift(1)) &
        (df['open'] < df['open'].shift(1)) &
        (df['low'].shift(1) > bear_gap_btm)
    )
    
    # Combined conditions for entries
    bull_fvg_valid = bull_fvg_cond.copy()
    bear_fvg_valid = bear_fvg_cond.copy()
    
    for i in range(len(df)):
        if i < 2:
            continue
        
        if bull_fvg_valid.iloc[i] and not pd.isna(df['high'].shift(2).iloc[i]):
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if bear_fvg_valid.iloc[i] and not pd.isna(df['low'].shift(2).iloc[i]):
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries