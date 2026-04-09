import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('datetime').sort_index()
    
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    df_4h = df.resample('4h').agg(agg_dict).dropna(how='all')
    df_4h = df_4h[df_4h['volume'] > 0]
    
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    close_4h = df_4h['close']
    volume_4h = df_4h['volume']
    
    volfilt = volume_4h.shift(1) > volume_4h.rolling(9).mean() * 1.5
    
    high_val = high_4h.values
    low_val = low_4h.values
    close_val = close_4h.values
    n = len(df_4h)
    tr = np.zeros(n)
    tr[0] = high_val[0] - low_val[0]
    for i in range(1, n):
        hl = high_val[i] - low_val[i]
        hpc = abs(high_val[i] - close_val[i-1])
        lpc = abs(low_val[i] - close_val[i-1])
        tr[i] = max(hl, hpc, lpc)
    atr_4h_raw = np.zeros(n)
    atr_4h_raw[0] = tr[0]
    alpha = 1/20
    for i in range(1, n):
        atr_4h_raw[i] = tr[i] * alpha + atr_4h_raw[i-1] * (1 - alpha)
    atr_4h = pd.Series(atr_4h_raw, index=df_4h.index)
    atr_threshold = atr_4h / 1.5
    atrfilt = ((low_4h - high_4h.shift(2) > atr_threshold) | (low_4h.shift(2) - high_4h > atr_threshold))
    
    sma_54 = close_4h.rolling(54).mean()
    loc2 = sma_54 > sma_54.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    bfvg = (low_4h > high_4h.shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (high_4h < low_4h.shift(2)) & volfilt & atrfilt & locfilts
    
    lastFVG = pd.Series(0, index=df_4h.index)
    lastFVG[bfvg] = 1
    lastFVG[sfvg] = -1
    lastFVG = lastFVG.replace(0, np.nan).ffill().fillna(0)
    
    bull_sharp = bfvg & (lastFVG.shift(1) == -1)
    bear_sharp = sfvg & (lastFVG.shift(1) == 1)
    
    df['hour'] = df.index.hour + df.index.minute / 60.0
    window1_start = 7 + 45/60
    window1_end = 11 + 45/60
    window2_start = 14 + 0/60
    window2_end = 14 + 45/60
    in_window = ((df['hour'] >= window1_start) & (df['hour'] < window1_end)) | ((df['hour'] >= window2_start) & (df['hour'] <= window2_end))
    
    df['period_4h'] = df.index.to_period('4h')
    df['is_first_4h'] = df['period_4h'] != df['period_4h'].shift(1)
    
    first_bar_of_4h = df[df['is_first_4h']].copy()
    
    entries = []
    trade_num = 1
    
    for idx, row in first_bar_of_4h.iterrows():
        period = row['period_4h']
        try:
            candle_4h_idx = df_4h.index[df_4h.index.to_period('4h') == period][0]
        except IndexError:
            continue
        
        candle_4h_loc = df_4h.index.get_loc(candle_4h_idx)
        if candle_4h_loc == 0:
            continue
        
        is_confirmed_and_new = True
        
        if is_confirmed_and_new and row['is_first_4h'] and in_window.get(idx, False):
            if bull_sharp.get(candle_4h_idx, False):
                ts = int(row['time'])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(row['close']),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(row['close']),
                    'raw_price_b': float(row['close'])
                })
                trade_num += 1
            elif bear_sharp.get(candle_4h_idx, False):
                ts = int(row['time'])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(row['close']),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(row['close']),
                    'raw_price_b': float(row['close'])
                })
                trade_num += 1
    
    return entries