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
    
    open_series = df['open']
    high_series = df['high']
    low_series = df['low']
    close_series = df['close']
    volume_series = df['volume']
    
    # Helper functions for candle patterns
    is_up = close_series > open_series
    is_down = close_series < open_series
    
    # OB detection (order block) - check previous bar
    is_ob_up = (is_down.shift(1)) & (is_up) & (close_series > high_series.shift(1))
    is_ob_down = (is_up.shift(1)) & (is_down) & (close_series < low_series.shift(1))
    
    # FVG detection (fair value gap) - check current and 2 bars back
    is_fvg_up = low_series > high_series.shift(2)
    is_fvg_down = high_series < low_series.shift(2)
    
    # Current bar OB and FVG states
    ob_up = is_ob_up.shift(1)
    ob_down = is_ob_down.shift(1)
    fvg_up = is_fvg_up
    fvg_down = is_fvg_down
    
    # Volume filter: volume[1] > sma(volume, 9) * 1.5
    vol_sma_9 = volume_series.rolling(9).mean()
    vol_filt = volume_series.shift(1) > vol_sma_9 * 1.5
    
    # ATR filter: (low - high[2] > atr) or (low[2] - high > atr)
    # Wilder ATR
    tr1 = high_series - low_series
    tr2 = (high_series - close_series.shift(1)).abs()
    tr3 = (low_series - close_series.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20 = tr.ewm(alpha=1/20, adjust=False).mean()
    atr_val = atr_20 / 1.5
    atr_filt = (low_series - high_series.shift(2) > atr_val) | (low_series.shift(2) - high_series > atr_val)
    
    # Trend filter using SMA
    loc = close_series.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    loc_filt_bull = loc2
    loc_filt_bear = ~loc2
    
    # Combined FVG conditions
    bfvg = (low_series > high_series.shift(2)) & vol_filt & atr_filt & loc_filt_bull
    sfvg = (high_series < low_series.shift(2)) & vol_filt & atr_filt & loc_filt_bear
    
    # Top Imbalance patterns
    top_imbalance_bway = (low_series.shift(2) <= open_series.shift(1)) & (high_series >= close_series.shift(1)) & (close_series < low_series.shift(1))
    top_imbalance_xbway = (low_series.shift(2) <= open_series.shift(1)) & (high_series >= close_series.shift(1)) & (close_series > low_series.shift(1))
    
    # Bottom Imbalance patterns
    bottom_imbalance_bway = (high_series.shift(2) >= open_series.shift(1)) & (low_series <= close_series.shift(1)) & (close_series > high_series.shift(1))
    bottom_imbalance_xbway = (high_series.shift(2) >= open_series.shift(1)) & (low_series <= close_series.shift(1)) & (close_series < high_series.shift(1))
    
    # Entry conditions - combination of OB and FVG patterns
    bull_entry_cond = (ob_up.shift(1)) & (fvg_up) & bfvg
    bear_entry_cond = (ob_down.shift(1)) & (fvg_down) & sfvg
    
    # Also include imbalance-based entries
    bull_entry_imbalance = top_imbalance_bway | bottom_imbalance_xbway
    bear_entry_imbalance = bottom_imbalance_bway | top_imbalance_xbway
    
    # Combine conditions
    long_cond = bull_entry_cond | bull_entry_imbalance
    short_cond = bear_entry_cond | bear_entry_imbalance
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i < 2:
            continue
        
        if pd.isna(close_series.iloc[i]) or pd.isna(high_series.iloc[i]) or pd.isna(low_series.iloc[i]):
            continue
        
        entry_price = close_series.iloc[i]
        
        if long_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
        
        elif short_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
    
    return entries