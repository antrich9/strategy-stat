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
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['4h_period'] = df['time_dt'].dt.to_period('4H')
    
    def get_4h_ohlcv(group):
        return pd.Series({
            'open': group.iloc[0]['open'],
            'high': group['high'].max(),
            'low': group['low'].min(),
            'close': group.iloc[-1]['close'],
            'volume': group['volume'].sum(),
            'time': group.iloc[0]['time']
        })
    
    df_4h = df.groupby('4h_period', sort=False).apply(get_4h_ohlcv).reset_index(drop=True)
    
    df_4h['volume_sma9'] = df_4h['volume'].rolling(9, min_periods=1).mean()
    df_4h['volfilt'] = df_4h['volume'].shift(1) > df_4h['volume_sma9'].shift(1) * 1.5
    
    atr_length = 20
    tr = pd.concat([
        df_4h['high'] - df_4h['low'],
        (df_4h['high'] - df_4h['close'].shift(1)).abs(),
        (df_4h['low'] - df_4h['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(window=atr_length, min_periods=atr_length).mean()
    atr_adjusted = atr / 1.5
    df_4h['atr_4h'] = atr_adjusted
    df_4h['atrfilt'] = ((df_4h['low'] - df_4h['high'].shift(2) > atr_adjusted) | 
                        (df_4h['low'].shift(2) - df_4h['high'] > atr_adjusted))
    
    df_4h['sma54'] = df_4h['close'].rolling(54, min_periods=54).mean()
    df_4h['trend_up'] = df_4h['sma54'] > df_4h['sma54'].shift(1)
    
    df_4h['bfvg'] = ((df_4h['low'] > df_4h['high'].shift(2)) & 
                     df_4h['volfilt'] & 
                     df_4h['atrfilt'] & 
                     df_4h['trend_up'])
    df_4h['sfvg'] = ((df_4h['high'] < df_4h['low'].shift(2)) & 
                     df_4h['volfilt'] & 
                     df_4h['atrfilt'] & 
                     ~df_4h['trend_up'])
    
    last_fvg = 0
    entries = []
    trade_num = 1
    
    for i in range(1, len(df_4h)):
        bfvg = df_4h['bfvg'].iloc[i]
        sfvg = df_4h['sfvg'].iloc[i]
        
        if bfvg and last_fvg == -1:
            ts = int(df_4h['time'].iloc[i])
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
            trade_num += 1
            last_fvg = 1
        elif sfvg and last_fvg == 1:
            ts = int(df_4h['time'].iloc[i])
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
            trade_num += 1
            last_fvg = -1
        elif bfvg:
            last_fvg = 1
        elif sfvg:
            last_fvg = -1
    
    return entries