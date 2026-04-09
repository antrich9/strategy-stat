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
    
    # Resample to 4H for FVG calculations
    df_4h = df.set_index('time')
    
    # Resample OHLCV to 4H
    ohlc_4h = {
        'open': df_4h['open'].resample('4H').first(),
        'high': df_4h['high'].resample('4H').max(),
        'low': df_4h['low'].resample('4H').min(),
        'close': df_4h['close'].resample('4H').last(),
        'volume': df_4h['volume'].resample('4H').sum()
    }
    
    df_4h = pd.DataFrame(ohlc_4h).dropna()
    
    # Calculate volume filter
    df_4h['vol_sma'] = df_4h['volume'].rolling(9).mean()
    df_4h['volfilt'] = df_4h['volume'].shift(1) > df_4h['vol_sma'] * 1.5
    
    # Calculate ATR filter
    high_low = df_4h['high'] - df_4h['low']
    high_close = np.abs(df_4h['high'] - df_4h['close'].shift(1))
    low_close = np.abs(df_4h['low'] - df_4h['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(20).mean() / 1.5
    df_4h['atrfilt'] = (df_4h['low'] - df_4h['high'].shift(2) > atr) | (df_4h['low'].shift(2) - df_4h['high'] > atr)
    
    # Calculate trend filter
    df_4h['loc1'] = df_4h['close'].rolling(54).mean()
    df_4h['loc21'] = df_4h['loc1'] > df_4h['loc1'].shift(1)
    df_4h['locfiltb'] = df_4h['loc21']
    df_4h['locfilts'] = ~df_4h['loc21']
    
    # Calculate FVGs
    df_4h['bfvg'] = (df_4h['low'] > df_4h['high'].shift(2)) & df_4h['volfilt'] & df_4h['atrfilt'] & df_4h['locfiltb']
    df_4h['sfvg'] = (df_4h['high'] < df_4h['low'].shift(2)) & df_4h['volfilt'] & df_4h['atrfilt'] & df_4h['locfilts']
    
    # Calculate sharp turns and detect entry conditions
    df_4h['prev_bfvg'] = df_4h['bfvg'].shift(1)
    df_4h['prev_sfvg'] = df_4h['sfvg'].shift(1)
    df_4h['sharp_turn_long'] = (df_4h['bfvg'] == True) & (df_4h['prev_sfvg'] == True)
    df_4h['sharp_turn_short'] = (df_4h['sfvg'] == True) & (df_4h['prev_bfvg'] == True)
    
    # Get the timestamp of sharp turns
    sharp_turn_ts = df_4h[df_4h['sharp_turn_long'] | df_4h['sharp_turn_short']].index
    
    # Map back to original dataframe and filter by time window
    entries = []
    for ts in sharp_turn_ts:
        if is_within_window(ts):
            entries.append(ts)
    
    return entries