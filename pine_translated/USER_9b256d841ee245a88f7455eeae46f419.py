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
    results = []
    trade_num = 0
    
    # Calculate ATR(200) using Wilder's method
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr4 = tr.ewm(alpha=1/200, adjust=False).mean()
    
    # Bullish Volume Imbalance detection
    bull_gap_top = np.minimum(df['close'], df['open'])
    bull_gap_btm = np.maximum(df['close'].shift(1), df['open'].shift(1))
    
    bull_vi_condition = (
        (df['open'] > df['close'].shift(1)) &
        (df['high'].shift(1) > df['low']) &
        (df['close'] > df['close'].shift(1)) &
        (df['open'] > df['open'].shift(1)) &
        (df['high'].shift(1) < bull_gap_top)
    )
    
    bull_vi = bull_vi_condition
    
    # Bearish Volume Imbalance detection
    bear_gap_top = np.minimum(df['close'].shift(1), df['open'].shift(1))
    bear_gap_btm = np.maximum(df['close'], df['open'])
    
    bear_vi_condition = (
        (df['open'] < df['close'].shift(1)) &
        (df['low'].shift(1) < df['high']) &
        (df['close'] < df['close'].shift(1)) &
        (df['open'] < df['open'].shift(1)) &
        (df['low'].shift(1) > bear_gap_btm)
    )
    
    bear_vi = bear_vi_condition
    
    # Bullish FVG detection
    bull_fvg_top = np.minimum(df['close'], df['open'])
    bull_fvg_btm = np.maximum(df['close'].shift(1), df['open'].shift(1))
    bull_fvg = (df['high'] < df['low'].shift(1)) & (df['low'] > bull_fvg_btm)
    
    # Bearish FVG detection
    bear_fvg_btm = np.maximum(df['close'], df['open'])
    bear_fvg_top = np.minimum(df['close'].shift(1), df['open'].shift(1))
    bear_fvg = (df['low'] > df['high'].shift(1)) & (df['high'] < bear_fvg_top)
    
    # Bullish OG detection
    bull_og_top = np.minimum(df['close'], df['open'])
    bull_og_btm = np.maximum(df['close'].shift(1), df['open'].shift(1))
    bull_og = (df['close'] > df['open'].shift(1)) & (df['close'] > df['open']) & (df['low'] > bull_og_btm)
    
    # Bearish OG detection
    bear_og_btm = np.maximum(df['close'], df['open'])
    bear_og_top = np.minimum(df['close'].shift(1), df['open'].shift(1))
    bear_og = (df['close'] < df['close'].shift(1)) & (df['close'] < df['open']) & (df['high'] < bear_og_top)
    
    # Long entry condition: Bullish imbalance filled OR bullish FVG present
    long_entry_cond = bull_vi | bull_fvg | bull_og
    
    # Short entry condition: Bearish imbalance filled OR bearish FVG present
    short_entry_cond = bear_vi | bear_fvg | bear_og
    
    # Skip first 2 bars due to shift dependencies
    for i in range(2, len(df)):
        if pd.isna(atr4.iloc[i]):
            continue
        
        # Long entry signals
        if long_entry_cond.iloc[i]:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            results.append({
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
        
        # Short entry signals
        if short_entry_cond.iloc[i]:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            results.append({
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
    
    return results