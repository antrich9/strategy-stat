import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['ts'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('ts').sort_index()
    
    # Resample to 4H for FVG detection
    high_4h = df['high'].resample('240min').max()
    low_4h = df['low'].resample('240min').min()
    close_4h = df['close'].resample('240min').last()
    volume_4h = df['volume'].resample('240min').sum()
    open_4h = df['open'].resample('240min').first()
    
    # Detect new 4H candles
    is_new_4h = pd.Series(True, index=open_4h.index)
    
    # Volume Filter (shifted by 1 to avoid lookahead)
    vol_avg = volume_4h.rolling(9).mean()
    volfilt = volume_4h.shift(1) > (vol_avg.shift(1) * 1.5)
    
    # ATR Filter - Wilder smoothing (length 20), shifted by 1
    tr1 = high_4h - low_4h
    tr2 = (high_4h - close_4h.shift(1)).abs()
    tr3 = (low_4h - close_4h.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/20, adjust=False).mean()
    atrfilt = ((low_4h.shift(1) - high_4h.shift(1).shift(1)) > (atr.shift(1) / 1.5)) | ((low_4h.shift(1).shift(1) - high_4h.shift(1)) > (atr.shift(1) / 1.5))
    
    # Trend Filter (SMA 54), shifted by 1
    loc = close_4h.rolling(54).mean()
    locfiltb = (loc.shift(1) > loc.shift(1).shift(1))
    locfilts = ~locfiltb
    
    # Detect FVGs (shifted by 1 to avoid lookahead)
    bfvg = (low_4h.shift(1) > high_4h.shift(1).shift(1)) & volfilt & atrfilt & locfiltb
    sfvg = (high_4h.shift(1) < low_4h.shift(1).shift(1)) & volfilt & atrfilt & locfilts
    
    # Track last FVG type for sharp turn detection
    lastFVG = 0
    entries = []
    
    # Iterate through 4H candles starting from index where indicators are valid
    for i in range(20, len(close_4h)):
        if is_new_4h.iloc[i]:
            if bfvg.iloc[i] and lastFVG == -1:
                lastFVG = 1
                entry_ts_4h = close_4h.index[i]
                # Find corresponding 15min bar
                idx_pos = df.index.get_indexer([entry_ts_4h], method='pad')[0]
                if idx_pos < len(df):
                    ts = df.index[idx_pos]
                    london_ts = ts + pd.Timedelta(hours=1)
                    hour, minute = london_ts.hour, london_ts.minute
                    is_london = (hour > 7 or (hour == 7 and minute >= 45)) and (hour < 17 or (hour == 17 and minute < 45))
                    if is_london:
                        entries.append((idx_pos, 'long', df['close'].iloc[idx_pos]))
            elif sfvg.iloc[i] and lastFVG == 1:
                lastFVG = -1
                entry_ts_4h = close_4h.index[i]
                idx_pos = df.index.get_indexer([entry_ts_4h], method='pad')[0]
                if idx_pos < len(df):
                    ts = df.index[idx_pos]
                    london_ts = ts + pd.Timedelta(hours=1)
                    hour, minute = london_ts.hour, london_ts.minute
                    is_london = (hour > 7 or (hour == 7 and minute >= 45)) and (hour < 17 or (hour == 17 and minute < 45))
                    if is_london:
                        entries.append((idx_pos, 'short', df['close'].iloc[idx_pos]))
            elif bfvg.iloc[i]:
                lastFVG = 1
            elif sfvg.iloc[i]:
                lastFVG = -1
    
    if not entries:
        return []
    
    result = []
    trade_num = 1
    for entry_idx, direction, price in entries:
        entry_ts = int(df['time'].iloc[entry_idx])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        result.append({
            'trade_num': trade_num,
            'direction': direction,
            'entry_ts': entry_ts,
            'entry_time': entry_time,
            'entry_price_guess': price,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': price,
            'raw_price_b': price
        })
        trade_num += 1
    
    return result