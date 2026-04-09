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
    
    # Set time as index and sort
    df = df.set_index('time').sort_index()
    
    # Resample to 4H candles to match Pine Script security function
    ohlc_4h = df.resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Volume filter (4H) - inp11 enabled by default
    vol_sma_9 = ohlc_4h['volume'].rolling(9).mean()
    volfilt1 = ohlc_4h['volume'].shift(1) > vol_sma_9 * 1.5
    
    # ATR filter (4H) - inp21 enabled by default
    atr_4h = calculate_atr(ohlc_4h, 20)
    atr_threshold = atr_4h / 1.5
    low_minus_high_2 = ohlc_4h['low'] - ohlc_4h['high'].shift(2)
    high_minus_low_2 = ohlc_4h['high'] - ohlc_4h['low'].shift(2)
    atrfilt1 = (low_minus_high_2 > atr_threshold) | (high_minus_low_2 > atr_threshold)
    
    # Trend filter (4H) - inp31 enabled by default
    sma_54 = ohlc_4h['close'].rolling(54).mean()
    loc2 = sma_54 > sma_54.shift(1)
    locfiltb1 = loc2
    locfilts1 = ~loc2
    
    # Bullish FVG detection (4H)
    bullish_fvg = (ohlc_4h['low'] > ohlc_4h['high'].shift(2)) & volfilt1 & atrfilt1 & locfiltb1
    
    # Bearish FVG detection (4H)
    bearish_fvg = (ohlc_4h['high'] < ohlc_4h['low'].shift(2)) & volfilt1 & atrfilt1 & locfilts1
    
    # Detect sharp turns - bullish FVG following bearish FVG, or vice versa
    prev_bullish_fvg = bullish_fvg.shift(1).fillna(False)
    prev_bearish_fvg = bearish_fvg.shift(1).fillna(False)
    
    bull_sharp_turn = bullish_fvg & prev_bearish_fvg
    bear_sharp_turn = bearish_fvg & prev_bullish_fvg
    
    # Combine and prepare entries
    entries = pd.DataFrame({
        'timestamp': ohlc_4h.index,
        'direction': np.where(bull_sharp_turn, 'long', 'short'),
        'price': ohlc_4h['close'].values
    })
    
    entries = entries[(bull_sharp_turn | bear_sharp_turn)].reset_index(drop=True)
    
    return entries.to_dict('records')