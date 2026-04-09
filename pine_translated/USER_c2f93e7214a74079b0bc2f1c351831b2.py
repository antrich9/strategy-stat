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
    
    inp11 = False
    inp21 = False
    inp31 = False
    atr_length1 = 20
    
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    def in_trading_window(dt):
        hour = dt.hour
        minute = dt.minute
        time_minutes = hour * 60 + minute
        in_w1 = 420 <= time_minutes < 705
        in_w2 = 840 <= time_minutes < 885
        return in_w1 or in_w2
    
    df['in_window'] = df['datetime'].apply(in_trading_window)
    
    df['time_4h'] = df['time'] // (14400) * 14400
    
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'in_window': 'max'
    }
    
    df_4h = df.groupby('time_4h').agg(agg_dict).reset_index()
    df_4h = df_4h.sort_values('time_4h').reset_index(drop=True)
    
    if inp11:
        vol_sma = df_4h['volume'].rolling(9).mean() * 1.5
        df_4h['volfilt1'] = df_4h['volume'].shift(1) > vol_sma
    else:
        df_4h['volfilt1'] = True
    
    high = df_4h['high']
    low = df_4h['low']
    close = df_4h['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/atr_length1, adjust=False).mean()
    df_4h['atr_4h_adj'] = atr / 1.5
    
    if inp21:
        df_4h['atrfilt1'] = ((low - high.shift(2) > df_4h['atr_4h_adj']) | (low.shift(2) - high > df_4h['atr_4h_adj']))
    else:
        df_4h['atrfilt1'] = True
    
    df_4h['loc1'] = close.rolling(54).mean()
    df_4h['loc21'] = df_4h['loc1'] > df_4h['loc1'].shift(1)
    
    if inp31:
        df_4h['locfiltb1'] = df_4h['loc21']
        df_4h['locfilts1'] = ~df_4h['loc21']
    else:
        df_4h['locfiltb1'] = True
        df_4h['locfilts1'] = True
    
    df_4h['bfvg1'] = (low > high.shift(2)) & df_4h['volfilt1'] & df_4h['atrfilt1'] & df_4h['locfiltb1']
    df_4h['sfvg1'] = (high < low.shift(2)) & df_4h['volfilt1'] & df_4h['atrfilt1'] & df_4h['locfilts1']
    
    lastFVG = 0
    entries = []
    trade_num = 1
    
    for i in range(1, len(df_4h)):
        row = df_4h.iloc[i]
        prev_row = df_4h.iloc[i-1]
        is_new_4h = row['time_4h'] != prev_row['time_4h']
        
        if is_new_4h:
            if row['bfvg1'] and lastFVG == -1:
                if row['in_window']:
                    ts = int(row['time_4h'])
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': ts,
                        'entry_time': dt.isoformat(),
                        'entry_price_guess': float(row['close']),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(row['close']),
                        'raw_price_b': float(row['close'])
                    })
                    trade_num += 1
                lastFVG = 1
            elif row['sfvg1'] and lastFVG == 1:
                if row['in_window']:
                    ts = int(row['time_4h'])
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': ts,
                        'entry_time': dt.isoformat(),
                        'entry_price_guess': float(row['close']),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(row['close']),
                        'raw_price_b': float(row['close'])
                    })
                    trade_num += 1
                lastFVG = -1
            elif row['bfvg1']:
                lastFVG = 1
            elif row['sfvg1']:
                lastFVG = -1
    
    return entries