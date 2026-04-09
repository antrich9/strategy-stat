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
    
    # Resample to 4H
    resampled = df.set_index(pd.to_datetime(df['time'], unit='s')).resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Recalculate indicators on 4H data
    vol_sma_4h = resampled['volume'].rolling(9).mean()
    vol_filter_4h = resampled['volume'].shift(1) > vol_sma_4h * 1.5
    
    high_4h = resampled['high']
    low_4h = resampled['low']
    close_4h = resampled['close']
    
    # Bullish FVG: low > high[2] and volfilt and atrfilt and locfiltb
    bull_fvg_4h = (low_4h > high_4h.shift(2)) & vol_filter_4h & atr_filter_4h & trend_filter_4h
    
    # Bearish FVG: high < low[2] and volfilt and atrfilt and locfilts
    bear_fvg_4h = (high_4h < low_4h.shift(2)) & vol_filter_4h & atr_filter_4h & ~trend_filter_4h
    
    # Get previous FVG states for sharp turn detection
    prev_bull_fvg_4h = bull_fvg_4h.shift(1).fillna(False)
    prev_bear_fvg_4h = bear_fvg_4h.shift(1).fillna(False)
    
    # Detect sharp turns: bullish FVG following bearish, or vice versa
    bull_sharp_turn = bull_fvg_4h & prev_bear_fvg_4h
    bear_sharp_turn = bear_fvg_4h & prev_bull_fvg_4h
    
    # Filter for new 4H candles only
    new_4h_mask = resampled.index.to_series().diff().dt.total_seconds() >= 14400
    
    # Apply confirmation and new candle filters
    valid_long_signals = bull_sharp_turn & new_4h_mask
    valid_short_signals = bear_sharp_turn & new_4h_mask
    
    # Merge with original dataframe
    df['bull_sharp_turn'] = np.where(valid_long_signals.reindex(df.index).fillna(False), 1, 0)
    df['bear_sharp_turn'] = np.where(valid_short_signals.reindex(df.index).fillna(False), 1, 0)
    
    return df[['bull_sharp_turn', 'bear_sharp_turn']].to_dict('records')