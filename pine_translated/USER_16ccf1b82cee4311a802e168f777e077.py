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
    
    df = df.copy()
    df['ts'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('ts', inplace=True)
    
    # Resample to 4H
    ohlc_4h = df[['open', 'high', 'low', 'close', 'volume']].resample('240T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    high_4h = ohlc_4h['high']
    low_4h = ohlc_4h['low']
    close_4h = ohlc_4h['close']
    volume_4h = ohlc_4h['volume']
    
    # Volume Filter
    volfilt1 = volume_4h.shift(1) > volume_4h.rolling(9).mean() * 1.5
    
    # ATR Filter (Wilder)
    tr1 = np.maximum(high_4h - low_4h, np.maximum(
        np.abs(high_4h - close_4h.shift(1)),
        np.abs(low_4h - close_4h.shift(1))
    ))
    atr_4h1 = tr1.ewm(alpha=1/20, adjust=False).mean() / 1.5
    atrfilt1 = (low_4h - high_4h.shift(2) > atr_4h1) | (low_4h.shift(2) - high_4h > atr_4h1)
    
    # Trend Filter
    loc1 = close_4h.rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    locfiltb1 = loc21
    locfilts1 = ~loc21
    
    # Bullish and Bearish FVGs
    bfvg1 = (low_4h > high_4h.shift(2)) & volfilt1 & atrfilt1 & locfiltb1
    sfvg1 = (high_4h < low_4h.shift(2)) & volfilt1 & atrfilt1 & locfilts1
    
    # Detect new 4H candle
    is_new_4h1 = ohlc_4h.index.to_series().diff() != pd.Timedelta(0)
    is_new_4h1.iloc[0] = True
    
    # Track last FVG type
    lastFVG = pd.Series(0.0, index=bfvg1.index)
    lastFVG[:] = 0.0
    
    for i in range(1, len(bfvg1)):
        if bfvg1.iloc[i]:
            lastFVG.iloc[i] = 1
        elif sfvg1.iloc[i]:
            lastFVG.iloc[i] = -1
        else:
            lastFVG.iloc[i] = lastFVG.iloc[i-1]
    
    # Detect sharp turns on confirmed new 4H candles
    bullish_sharp_turn = bfvg1 & (lastFVG.shift(1) == -1)
    bearish_sharp_turn = sfvg1 & (lastFVG.shift(1) == 1)
    sharp_turn_signal = bullish_sharp_turn | bearish_sharp_turn
    
    # Only on confirmed new 4H candles
    entry_condition = sharp_turn_signal & is_new_4h1
    
    # Map 4H signals back to original timeframe
    df_reset = df.reset_index()
    ohlc_4h_reset = ohlc_4h.reset_index()
    ohlc_4h_reset['entry_ts_4h'] = ohlc_4h_reset['ts']
    
    entry_condition_4h = entry_condition.reset_index(drop=True)
    ohlc_4h_reset['entry_cond'] = entry_condition_4h.values
    ohlc_4h_reset['direction'] = np.where(bullish_sharp_turn.reset_index(drop=True).values, 'long', 'short')
    
    entry_bars = ohlc_4h_reset[ohlc_4h_reset['entry_cond']]
    
    results = []
    trade_num = 1
    
    for _, row in entry_bars.iterrows():
        entry_ts = int(row['ts'].timestamp())
        entry_time = row['ts'].isoformat()
        entry_price_guess = float(row['close'])
        direction = row['direction']
        
        results.append({
            'trade_num': trade_num,
            'direction': direction,
            'entry_ts': entry_ts,
            'entry_time': entry_time,
            'entry_price_guess': entry_price_guess,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': entry_price_guess,
            'raw_price_b': entry_price_guess
        })
        trade_num += 1
    
    return results