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
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    
    # Resample to 4H for HTF data
    htf_4h = df[['open', 'high', 'low', 'close', 'volume']].resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Shifted values for FVG detection (avoid repainting)
    high_4h_prev = htf_4h['high'].shift(1)
    low_4h_prev = htf_4h['low'].shift(1)
    high_4h_prev2 = htf_4h['high'].shift(2)
    low_4h_prev2 = htf_4h['low'].shift(2)
    close_4h_prev = htf_4h['close'].shift(1)
    
    # Volume filter
    vol_sma = htf_4h['volume'].rolling(9).mean()
    vol_filter = htf_4h['volume'] > vol_sma * 1.5
    
    # ATR filter
    tr1 = htf_4h['high'] - htf_4h['low']
    tr2 = (htf_4h['high'] - htf_4h['close'].shift(1)).abs()
    tr3 = (htf_4h['low'] - htf_4h['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3]).resample('4H').max()
    atr_4h = tr.ewm(alpha=1/14, adjust=False).mean().reindex(htf_4h.index)
    atr_threshold = atr_4h / 1.5
    atrfilt = ((htf_4h['low'] - high_4h_prev2 > atr_threshold) | (low_4h_prev2 - htf_4h['high'] > atr_threshold))
    
    # Trend filter
    loc1 = htf_4h['close'].rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    locfiltb = loc21
    locfilts = ~loc21
    
    # FVG conditions
    bull_fvg_4h = (htf_4h['low'] > high_4h_prev2) & vol_filter & atrfilt & locfiltb
    bear_fvg_4h = (htf_4h['high'] < low_4h_prev2) & vol_filter & atrfilt & locfilts
    
    # Reindex to 15M timeframe
    bull_fvg_reindexed = bull_fvg_4h.reindex(df.index, method='ffill').fillna(False)
    bear_fvg_reindexed = bear_fvg_4h.reindex(df.index, method='ffill').fillna(False)
    
    entries = []
    trade_num = 1
    prev_4h_fvg = 0
    
    for i in range(1, len(df)):
        curr_bull = bull_fvg_reindexed.iloc[i]
        curr_bear = bear_fvg_reindexed.iloc[i]
        
        if curr_bull and prev_4h_fvg == -1:
            entry_time = int(df.index[i].timestamp())
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_time,
                'entry_time': datetime.fromtimestamp(entry_time, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            prev_4h_fvg = 1
        elif curr_bear and prev_4h_fvg == 1:
            entry_time = int(df.index[i].timestamp())
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_time,
                'entry_time': datetime.fromtimestamp(entry_time, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            prev_4h_fvg = -1
        elif curr_bull:
            prev_4h_fvg = 1
        elif curr_bear:
            prev_4h_fvg = -1
    
    return entries