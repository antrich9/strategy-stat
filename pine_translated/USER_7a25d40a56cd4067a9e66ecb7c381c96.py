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
    
    # Settings
    inp11 = False
    inp21 = False
    inp31 = False
    
    atr_length1 = 20
    lookback_bars = 12
    threshold = 0.0
    
    # Resample to 4H
    df_4h = df.set_index('time').resample('240T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    
    df_4h.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    
    # Detect new 4H candle
    df_4h['is_new_4h'] = True
    
    # 4H Volume filter
    volfilt1_series = df_4h['volume'].rolling(9).mean() * 1.5
    volfilt1 = df_4h['volume'].shift(1) > volfilt1_series.shift(1)
    
    # Wilder RSI implementation
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    # Wilder ATR implementation
    def wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/length, adjust=False).mean()
        return atr
    
    # 4H ATR
    atr_4h1_series = wilder_atr(df_4h['high'], df_4h['low'], df_4h['close'], atr_length1)
    atr_4h1 = atr_4h1_series / 1.5
    
    # 4H ATR Filter
    atrfilt1 = (df_4h['low'] - df_4h['high'].shift(2) > atr_4h1) | (df_4h['low'].shift(2) - df_4h['high'] > atr_4h1)
    
    # 4H Trend Filter
    loc1 = df_4h['close'].rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    locfiltb1 = loc21 if inp31 else pd.Series(True, index=loc21.index)
    locfilts1 = ~loc21 if inp31 else pd.Series(True, index=loc21.index)
    
    # Bullish and Bearish FVGs on 4H
    bfvg1 = (df_4h['low'] > df_4h['high'].shift(2)) & (volfilt1 if inp11 else pd.Series(True, index=volfilt1.index)) & (atrfilt1 if inp21 else pd.Series(True, index=atrfilt1.index)) & (locfiltb1 if inp31 else pd.Series(True, index=locfiltb1.index))
    sfvg1 = (df_4h['high'] < df_4h['low'].shift(2)) & (volfilt1 if inp11 else pd.Series(True, index=volfilt1.index)) & (atrfilt1 if inp21 else pd.Series(True, index=atrfilt1.index)) & (locfilts1 if inp31 else pd.Series(True, index=locfilts1.index))
    
    # Sharp Turn Detection (entries on new 4H candles)
    lastFVG = 0
    trade_num = 0
    entries = []
    
    for i in range(1, len(df_4h)):
        if not df_4h['is_new_4h'].iloc[i]:
            continue
        
        bfvg_current = bfvg1.iloc[i] if not pd.isna(bfvg1.iloc[i]) else False
        sfvg_current = sfvg1.iloc[i] if not pd.isna(sfvg1.iloc[i]) else False
        
        # Bullish Sharp Turn
        if bfvg_current and lastFVG == -1:
            trade_num += 1
            ts = int(df_4h['time'].iloc[i].timestamp())
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df_4h['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df_4h['close'].iloc[i],
                'raw_price_b': df_4h['close'].iloc[i]
            })
            lastFVG = 1
        # Bearish Sharp Turn
        elif sfvg_current and lastFVG == 1:
            trade_num += 1
            ts = int(df_4h['time'].iloc[i].timestamp())
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df_4h['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df_4h['close'].iloc[i],
                'raw_price_b': df_4h['close'].iloc[i]
            })
            lastFVG = -1
        elif bfvg_current:
            lastFVG = 1
        elif sfvg_current:
            lastFVG = -1
    
    return entries