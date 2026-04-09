import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    entries = []
    trade_num = 1
    
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    else:
        df['datetime'] = df['time']
    
    daily = df.resample('D', on='datetime').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    daily['dailyHigh21'] = daily['high'].shift(1)
    daily['dailyHigh22'] = daily['high'].shift(2)
    daily['dailyHigh11_3'] = daily['high'].shift(3)
    daily['dailyHigh11_4'] = daily['high'].shift(4)
    daily['dailyLow21'] = daily['low'].shift(1)
    daily['dailyLow22'] = daily['low'].shift(2)
    daily['dailyLow11_3'] = daily['low'].shift(3)
    daily['dailyLow11_4'] = daily['low'].shift(4)
    
    daily['is_swing_high'] = (daily['dailyHigh21'] < daily['dailyHigh22']) & \
                              (daily['dailyHigh11_3'] < daily['dailyHigh22']) & \
                              (daily['dailyHigh11_4'] < daily['dailyHigh22'])
    daily['is_swing_low'] = (daily['dailyLow21'] > daily['dailyLow22']) & \
                            (daily['dailyLow11_3'] > daily['dailyLow22']) & \
                            (daily['dailyLow11_4'] > daily['dailyLow22'])
    
    last_swing_type = "none"
    for idx in daily.index:
        if daily.loc[idx, 'is_swing_high']:
            last_swing_type = "dailyHigh"
        elif daily.loc[idx, 'is_swing_low']:
            last_swing_type = "dailyLow"
        daily.loc[idx, 'last_swing_type'] = last_swing_type
    
    df['date'] = df['datetime'].dt.date
    daily['date'] = daily.index.date
    last_swing_map = daily.set_index('date')['last_swing_type']
    df['last_swing_type'] = df['date'].map(last_swing_map)
    
    df['datetime_4h'] = df['datetime'].dt.floor('4h')
    h4 = df.groupby('datetime_4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    h4['high_4h'] = h4['high']
    h4['high_4h_2'] = h4['high'].shift(2)
    h4['low_4h'] = h4['low']
    h4['low_4h_2'] = h4['low'].shift(2)
    
    h4['bfvg'] = h4['low_4h'] > h4['high_4h_2']
    h4['sfvg'] = h4['high_4h'] < h4['low_4h_2']
    
    h4['vol_sma'] = h4['volume'].rolling(9).mean()
    h4['vol_filter'] = h4['volume'].shift(1) > h4['vol_sma'] * 1.5
    
    h4['tr'] = np.maximum(h4['high'] - h4['low'],
                         np.maximum(abs(h4['high'] - h4['close'].shift(1)),
                                   abs(h4['low'] - h4['close'].shift(1))))
    h4['atr'] = h4['tr'].ewm(com=19, adjust=False).mean() / 1.5
    h4['atr_filter'] = ((h4['low_4h'] - h4['high_4h_2'] > h4['atr']) | 
                       (h4['low_4h'].shift(2) - h4['high_4h'] > h4['atr']))
    
    h4['sma54'] = h4['close'].rolling(54).mean()
    h4['trend_up'] = h4['sma54'] > h4['sma54'].shift(1)
    
    df['bfvg'] = df['datetime_4h'].map(h4['bfvg'])
    df['sfvg'] = df['datetime_4h'].map(h4['sfvg'])
    df['vol_filter'] = df['datetime_4h'].map(h4['vol_filter'])
    df['atr_filter'] = df['datetime_4h'].map(h4['atr_filter'])
    df['trend_up'] = df['datetime_4h'].map(h4['trend_up'])
    
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    
    window1 = ((df['hour'] == 7) & (df['minute'] >= 45)) | \
              ((df['hour'] >= 8) & (df['hour'] < 11)) | \
              ((df['hour'] == 11) & (df['minute'] < 45))
    window2 = (df['hour'] == 14) & (df['minute'] <= 45)
    
    df['in_window'] = window1 | window2
    
    long_condition = (df['bfvg'] == True) & \
                     (df['last_swing_type'] == 'dailyLow') & \
                     (df['vol_filter'] == True) & \
                     (df['atr_filter'] == True) & \
                     (df['trend_up'] == True) & \
                     (df['in_window'] == True)
    
    short_condition = (df['sfvg'] == True) & \
                      (df['vol_filter'] == True) & \
                      (df['atr_filter'] == True) & \
                      (df['trend_up'] == False) & \
                      (df['in_window'] == True)
    
    for i in range(len(df)):
        if long_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': df['datetime'].iloc[i].isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': df['datetime'].iloc[i].isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries