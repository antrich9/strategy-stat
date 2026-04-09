import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('datetime').sort_index()
    
    resampled = df.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().copy()
    
    if len(resampled) < 3:
        return []
    
    resampled['vol_sma'] = resampled['volume'].rolling(9).mean()
    resampled['volfilt'] = resampled['volume'] > resampled['vol_sma'] * 1.5
    
    atr_length = 20
    tr1 = resampled['high'] - resampled['low']
    tr2 = abs(resampled['high'] - resampled['close'].shift(1))
    tr3 = abs(resampled['low'] - resampled['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    alpha = 1 / atr_length
    resampled['atr'] = tr.ewm(alpha=alpha, adjust=False).mean() / 1.5
    
    resampled['sma54'] = resampled['close'].rolling(54).mean()
    resampled['trend_up'] = resampled['sma54'] > resampled['sma54'].shift(1)
    
    resampled['atr_gap_up'] = resampled['low'] - resampled['high'].shift(2) > resampled['atr']
    resampled['atr_gap_down'] = resampled['low'].shift(2) - resampled['high'] > resampled['atr']
    
    bull_fvg = (resampled['low'] > resampled['high'].shift(2)) & \
               resampled['volfilt'] & \
               resampled['atr_gap_up'] & \
               resampled['trend_up']
    
    bear_fvg = (resampled['high'] < resampled['low'].shift(2)) & \
               resampled['volfilt'] & \
               resampled['atr_gap_down'] & \
               (~resampled['trend_up'])
    
    resampled['bull_fvg'] = bull_fvg
    resampled['bear_fvg'] = bear_fvg
    
    entries = []
    trade_num = 0
    last_fvg = 0
    
    for i in range(1, len(resampled)):
        row = resampled.iloc[i]
        
        if pd.isna(row.get('vol_sma')) or pd.isna(row.get('atr')) or pd.isna(row.get('sma54')):
            continue
        
        curr_fvg = 0
        if row['bull_fvg']:
            curr_fvg = 1
        elif row['bear_fvg']:
            curr_fvg = -1
        
        if curr_fvg != 0:
            last_fvg = curr_fvg
        
        candle_time = resampled.index[i]
        hour = candle_time.hour
        minute = candle_time.minute
        time_in_minutes = hour * 60 + minute
        london_start = 7 * 60 + 45
        london_end = 17 * 60 + 45
        is_within_time_window = (time_in_minutes >= london_start) & (time_in_minutes < london_end)
        
        if not is_within_time_window:
            continue
        
        if curr_fvg == 1 and last_fvg == -1:
            trade_num += 1
            entry_ts = int(resampled.index[i].timestamp())
            entry_time_str = resampled.index[i].isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': row['close'],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': row['close'],
                'raw_price_b': row['close']
            })
        elif curr_fvg == -1 and last_fvg == 1:
            trade_num += 1
            entry_ts = int(resampled.index[i].timestamp())
            entry_time_str = resampled.index[i].isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': row['close'],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': row['close'],
                'raw_price_b': row['close']
            })
    
    return entries