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
    
    # Higher timeframe settings
    higher_tf_1 = '240'  # 4H
    higher_tf_2 = '60'   # 1H
    higher_tf_3 = '15'   # 15min
    
    # Aggregate to higher timeframes using resample
    tf_configs = [
        ('240', higher_tf_1),
        ('60', higher_tf_2),
        ('15', higher_tf_3)
    ]
    
    htf_data = {}
    for name, tf in tf_configs:
        resampled = df.set_index('time').resample(tf).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        htf_data[tf] = resampled
    
    htf_close_1 = htf_data[higher_tf_1]['close']
    htf_open_1 = htf_data[higher_tf_1]['open']
    htf_high_1 = htf_data[higher_tf_1]['high']
    htf_low_1 = htf_data[higher_tf_1]['low']
    htf_volume_1 = htf_data[higher_tf_1]['volume']
    
    htf_close_2 = htf_data[higher_tf_2]['close']
    htf_open_2 = htf_data[higher_tf_2]['open']
    
    htf_close_3 = htf_data[higher_tf_3]['close']
    htf_open_3 = htf_data[higher_tf_3]['open']
    
    # Volume Filter (disabled by default)
    volfilt1 = htf_volume_1.shift(1) > htf_volume_1.rolling(9).mean() * 1.5
    
    # ATR Filter (disabled by default)
    atr_length1 = 20
    high_low = htf_high_1 - htf_low_1
    high_close = np.abs(htf_high_1 - htf_close_1.shift(1))
    low_close = np.abs(htf_low_1 - htf_close_1.shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_4h1 = tr.ewm(alpha=1.0/atr_length1, adjust=False).mean() / 1.5
    atrfilt1 = (htf_low_1 - htf_high_1.shift(2) > atr_4h1) | (htf_low_1.shift(2) - htf_high_1 > atr_4h1)
    
    # Trend Filter (disabled by default)
    loc1 = htf_close_1.rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    locfiltb1 = loc21
    locfilts1 = ~loc21
    
    # Identify Bullish and Bearish FVGs
    bfvg1 = (htf_low_1 > htf_high_1.shift(2)) & volfilt1 & atrfilt1 & locfiltb1
    sfvg1 = (htf_high_1 < htf_low_1.shift(2)) & volfilt1 & atrfilt1 & locfilts1
    
    # Time windows for London sessions
    london_start_morning = df['time'].dt.tz_localize(None).dt.floor('D') + pd.Timedelta(hours=8, minutes=45)
    london_end_morning = df['time'].dt.tz_localize(None).dt.floor('D') + pd.Timedelta(hours=9, minutes=45)
    london_start_afternoon = df['time'].dt.tz_localize(None).dt.floor('D') + pd.Timedelta(hours=13, minutes=45)
    london_end_afternoon = df['time'].dt.tz_localize(None).dt.floor('D') + pd.Timedelta(hours=15, minutes=45)
    
    london_start_morning = london_start_morning.tz_localize('UTC')
    london_end_morning = london_end_morning.tz_localize('UTC')
    london_start_afternoon = london_start_afternoon.tz_localize('UTC')
    london_end_afternoon = london_end_afternoon.tz_localize('UTC')
    
    isWithinMorningWindow = (df['time'] >= london_start_morning) & (df['time'] < london_end_morning)
    isWithinAfternoonWindow = (df['time'] >= london_start_afternoon) & (df['time'] < london_end_afternoon)
    isWithinTimeWindow = isWithinMorningWindow | isWithinAfternoonWindow
    
    # Detect new 4H candle
    htf_time_1 = htf_data[higher_tf_1].index
    is_new_4h1 = pd.Series(False, index=df.index)
    for i, row_time in enumerate(df['time']):
        for j in range(len(htf_time_1) - 1):
            if htf_time_1[j] <= row_time < htf_time_1[j + 1]:
                if j > 0 and htf_time_1[j - 1] < row_time:
                    continue
                if row_time == htf_time_1[j]:
                    is_new_4h1.iloc[i] = True
                break
    
    # Find matching 4H indices for FVGs
    bfvg1_reindexed = bfvg1.reindex(df['time']).ffill().fillna(False)
    sfvg1_reindexed = sfvg1.reindex(df['time']).ffill().fillna(False)
    
    entries = []
    trade_num = 1
    lastFVG = 0
    
    for i in range(1, len(df)):
        if not is_new_4h1.iloc[i]:
            continue
        
        current_bfvg = bfvg1_reindexed.iloc[i]
        current_sfvg = sfvg1_reindexed.iloc[i]
        
        if current_bfvg and lastFVG == -1:
            ts = int(df['time'].iloc[i].timestamp())
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            lastFVG = 1
        elif current_sfvg and lastFVG == 1:
            ts = int(df['time'].iloc[i].timestamp())
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            lastFVG = -1
        elif current_bfvg:
            lastFVG = 1
        elif current_sfvg:
            lastFVG = -1
    
    return entries