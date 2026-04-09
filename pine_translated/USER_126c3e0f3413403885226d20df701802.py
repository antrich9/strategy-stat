import pandas as pd
import numpy as np
from datetime import datetime, timezone

def calculate_atr(high, low, close, length):
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(np.nan, index=tr.index)
    atr.iloc[length - 1] = tr.iloc[:length].sum()
    for i in range(length, len(tr)):
        atr.iloc[i] = (atr.iloc[i - 1] * (length - 1) + tr.iloc[i]) / length
    return atr

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('time_dt')
    
    o = df['open'].resample('240min').first()
    h = df['high'].resample('240min').max()
    l = df['low'].resample('240min').min()
    c = df['close'].resample('240min').last()
    v = df['volume'].resample('240min').sum()
    
    df_4h = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})
    df_4h = df_4h.dropna().reset_index()
    df_4h = df_4h.rename(columns={'time_dt': 'timestamp', 'index': 'timestamp'})
    df_4h['timestamp'] = df_4h['timestamp'].dt.tz_localize(None)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_vals = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    high_4h_s = pd.Series(high_4h)
    low_4h_s = pd.Series(low_4h)
    close_4h_s = pd.Series(close_4h_vals)
    volume_4h_s = pd.Series(volume_4h)
    
    volfilt = volume_4h_s.shift(1) > pd.Series(volume_4h).rolling(9).mean().shift(1) * 1.5
    
    atr_4h = calculate_atr(high_4h_s, low_4h_s, close_4h_s, 20)
    atr_4h_filt = atr_4h.shift(1) / 1.5
    atrfilt = ((low_4h_s - high_4h_s.shift(2) > atr_4h_filt) | (low_4h_s.shift(2) - high_4h_s > atr_4h_filt))
    
    loc1 = close_4h_s.rolling(54).mean()
    loc2 = loc1 > loc1.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    bull_fvg = (low_4h_s > high_4h_s.shift(2)) & volfilt & atrfilt & locfiltb
    bear_fvg = (high_4h_s < low_4h_s.shift(2)) & volfilt & atrfilt & locfilts
    
    entries = []
    trade_num = 1
    last_fvg = 0
    
    for i in range(2, len(df_4h)):
        if bull_fvg.iloc[i] and last_fvg == -1:
            ts = int(df_4h['timestamp'].iloc[i].timestamp())
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': close_4h_vals[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_4h_vals[i],
                'raw_price_b': close_4h_vals[i]
            })
            trade_num += 1
        elif bear_fvg.iloc[i] and last_fvg == 1:
            ts = int(df_4h['timestamp'].iloc[i].timestamp())
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': close_4h_vals[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close_4h_vals[i],
                'raw_price_b': close_4h_vals[i]
            })
            trade_num += 1
        
        if bull_fvg.iloc[i]:
            last_fvg = 1
        elif bear_fvg.iloc[i]:
            last_fvg = -1
    
    return entries