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
    resample_freq = '240min'
    tf_df = df.set_index('time')
    tf_df.index = pd.to_datetime(tf_df.index, unit='s', utc=True).tz_convert('UTC')
    tf_df = tf_df.resample(resample_freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    
    tf_df['dailyHigh'] = tf_df['high']
    tf_df['dailyLow'] = tf_df['low']
    tf_df['dailyHigh1'] = tf_df['dailyHigh'].shift(1)
    tf_df['dailyLow1'] = tf_df['dailyLow'].shift(1)
    tf_df['dailyHigh2'] = tf_df['dailyHigh'].shift(2)
    tf_df['dailyLow2'] = tf_df['dailyLow'].shift(2)
    
    inp1 = False
    inp2 = False
    inp3 = False
    
    volfilt = tf_df['volume'].shift(1) > tf_df['volume'].rolling(9).mean() * 1.5
    
    high_low = tf_df['high'] - tf_df['low']
    high_close_prev = np.abs(tf_df['high'] - tf_df['close'].shift(1))
    low_close_prev = np.abs(tf_df['low'] - tf_df['close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    atr = pd.Series(tr).ewm(alpha=1.0/20, adjust=False).mean()
    tf_df['atr2'] = atr / 1.5
    atrfilt = ((tf_df['dailyLow'] - tf_df['dailyHigh2'] > tf_df['atr2']) | (tf_df['dailyLow2'] - tf_df['dailyHigh'] > tf_df['atr2']))
    
    loc = tf_df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    tf_df['volfilt'] = volfilt
    tf_df['atrfilt'] = atrfilt
    tf_df['locfiltb'] = locfiltb
    tf_df['locfilts'] = locfilts
    tf_df['bfvg'] = (tf_df['dailyLow'] > tf_df['dailyHigh2']) & tf_df['volfilt'] & tf_df['atrfilt'] & tf_df['locfiltb']
    tf_df['sfvg'] = (tf_df['dailyHigh'] < tf_df['dailyLow2']) & tf_df['volfilt'] & tf_df['atrfilt'] & tf_df['locfilts']
    tf_df['bullishFVG'] = tf_df['bfvg']
    tf_df['bearishFVG'] = tf_df['sfvg']
    
    entries = []
    trade_num = 1
    
    for i in range(len(tf_df)):
        if pd.isna(tf_df['bfvg'].iloc[i]) or pd.isna(tf_df['sfvg'].iloc[i]):
            continue
        if tf_df['bullishFVG'].iloc[i]:
            entry_ts = int(tf_df['time'].iloc[i].timestamp())
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': tf_df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': tf_df['close'].iloc[i],
                'raw_price_b': tf_df['close'].iloc[i]
            })
            trade_num += 1
        if tf_df['bearishFVG'].iloc[i]:
            entry_ts = int(tf_df['time'].iloc[i].timestamp())
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': tf_df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': tf_df['close'].iloc[i],
                'raw_price_b': tf_df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries