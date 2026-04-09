import pandas as pd
import numpy as np
from datetime import datetime, timezone

def calculate_wilder_atr(high, low, close, length):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def generate_entries(df: pd.DataFrame) -> list:
    entries = []
    trade_num = 1
    
    # Store original columns to restore later
    orig_cols = df.columns.tolist()
    
    # Volume filter: inp1 = false by default, so volfilt = true always
    df['vol_sma'] = df['volume'].rolling(9).mean()
    df['vol_filt'] = df['volume'].shift(1) > df['vol_sma'] * 1.5
    
    # ATR filter: inp2 = false by default, so atrfilt = true always
    df['atr'] = calculate_wilder_atr(df['high'], df['low'], df['close'], 20)
    df['atr2'] = df['atr'] / 1.5
    df['atr_filt'] = ((df['low'] - df['high'].shift(2) > df['atr2']) | (df['low'].shift(2) - df['high'] > df['atr2']))
    
    # Trend filter: inp3 = false by default, so locfiltb = locfilts = true always
    df['loc'] = df['close'].rolling(54).mean()
    df['loc_trend'] = df['loc'] > df['loc'].shift(1)
    df['locfiltb'] = df['loc_trend']
    df['locfilts'] = ~df['loc_trend']
    
    # Bullish FVG: low > high[2] and volfilt and atrfilt and locfiltb
    df['bfvg'] = (df['low'] > df['high'].shift(2)) & df['vol_filt'] & df['atr_filt'] & df['locfiltb']
    
    # Bearish FVG: high < low[2] and volfilt and atrfilt and locfilts
    df['sfvg'] = (df['high'] < df['low'].shift(2)) & df['vol_filt'] & df['atr_filt'] & df['locfilts']
    
    for i in range(len(df)):
        if df['bfvg'].iloc[i]:
            ts = int(df['time'].iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            }
            entries.append(entry)
            trade_num += 1
        elif df['sfvg'].iloc[i]:
            ts = int(df['time'].iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            }
            entries.append(entry)
            trade_num += 1
    
    # Remove temporary columns
    temp_cols = ['vol_sma', 'vol_filt', 'atr', 'atr2', 'atr_filt', 'loc', 'loc_trend', 'locfiltb', 'locfilts', 'bfvg', 'sfvg']
    for col in temp_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    
    return entries