import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    
    # Resample to daily to get daily OHLC values
    df['date'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.date
    daily = df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    daily = daily.rename(columns={'high': 'dh', 'low': 'dl', 'close': 'dc', 'open': 'do_'})
    
    # Create daily shifted values
    daily['dailyHigh11'] = daily['dh'].shift(1)
    daily['dailyLow11'] = daily['dl'].shift(1)
    daily['dailyHigh21'] = daily['dh'].shift(1)
    daily['dailyLow21'] = daily['dl'].shift(1)
    daily['dailyHigh22'] = daily['dh'].shift(2)
    daily['dailyLow22'] = daily['dl'].shift(2)
    daily['prevDayHigh11'] = daily['dh'].shift(1)
    daily['prevDayLow11'] = daily['dl'].shift(1)
    
    # Merge daily data back
    df = df.merge(daily[['date', 'dailyHigh11', 'dailyLow11', 'dailyHigh21', 'dailyLow21', 
                         'dailyHigh22', 'dailyLow22', 'prevDayHigh11', 'prevDayLow11']], on='date', how='left')
    
    # Volume filter: volume[1] > sma(volume, 9) * 1.5
    df['vol_sma9'] = df['volume'].shift(1).rolling(9).mean()
    df['volfilt11'] = df['volume'].shift(1) > df['vol_sma9'] * 1.5
    
    # ATR filter: (dailyLow22 - dailyHigh22 > atr211) or (dailyLow22 - dailyHigh11 > atr211)
    # Using Wilder ATR manually
    tr = np.maximum(df['high'] - df['low'], 
                    np.maximum(np.abs(df['high'] - df['close'].shift(1)),
                               np.abs(df['low'] - df['close'].shift(1))))
    atr211 = tr.ewm(alpha=1/20, adjust=False).mean() / 1.5
    
    df['dailyLow22_filled'] = df['dailyLow22'].fillna(method='ffill')
    df['dailyHigh22_filled'] = df['dailyHigh22'].fillna(method='ffill')
    df['dailyHigh11_filled'] = df['dailyHigh11'].fillna(method='ffill')
    
    df['atr211'] = atr211
    df['atrfilt11'] = (df['dailyLow22_filled'] - df['dailyHigh22_filled'] > df['atr211']) | \
                      (df['dailyLow22_filled'] - df['dailyHigh11_filled'] > df['atr211'])
    
    # Trend filter using SMA(close, 54)
    df['loc11'] = df['close'].ewm(span=54, adjust=False).mean()
    df['loc211'] = df['loc11'] > df['loc11'].shift(1)
    df['locfiltb11'] = df['loc211']
    df['locfilts11'] = ~df['loc211']
    
    # Swing detection
    df['is_swing_high11'] = (df['dailyHigh21'] < df['dailyHigh22']) & \
                            (df['dailyHigh11'].shift(3) < df['dailyHigh22']) & \
                            (df['dailyHigh11'].shift(4) < df['dailyHigh22'])
    df['is_swing_low11'] = (df['dailyLow21'] > df['dailyLow22']) & \
                           (df['dailyLow11'].shift(3) > df['dailyLow22']) & \
                           (df['dailyLow11'].shift(4) > df['dailyLow22'])
    
    # Last swing type tracking
    last_swing_high = np.nan
    last_swing_low = np.nan
    lastSwingType = 'none'
    
    swing_high_arr = []
    swing_low_arr = []
    lastSwingType_arr = []
    
    for i in range(len(df)):
        sh = df['is_swing_high11'].iloc[i]
        sl = df['is_swing_low11'].iloc[i]
        
        if sh and not pd.isna(df['dailyHigh22'].iloc[i]):
            last_swing_high = df['dailyHigh22'].iloc[i]
            lastSwingType = 'dailyHigh'
        if sl and not pd.isna(df['dailyLow22'].iloc[i]):
            last_swing_low = df['dailyLow22'].iloc[i]
            lastSwingType = 'dailyLow'
        
        swing_high_arr.append(last_swing_high)
        swing_low_arr.append(last_swing_low)
        lastSwingType_arr.append(lastSwingType)
    
    df['last_swing_high11'] = swing_high_arr
    df['last_swing_low11'] = swing_low_arr
    df['lastSwingType11'] = lastSwingType_arr
    
    # Trading windows (Europe/London timezone)
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Europe/London')
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    
    def in_window(row):
        h, m = row['hour'], row['minute']
        # Window 1: 07:00-11:45
        w1 = (h == 7 and m >= 0) or (7 < h < 11) or (h == 11 and m <= 45)
        # Window 2: 14:00-14:45
        w2 = (h == 14 and m >= 0) or (h == 14 and m <= 45)
        return w1 or w2
    
    df['in_trading_window'] = df.apply(in_window, axis=1)
    
    # FVG conditions
    df['bfvg11'] = (df['dailyLow11'] > df['dailyHigh22']) & df['volfilt11'] & df['atrfilt11'] & df['locfiltb11']
    df['sfvg11'] = (df['dailyHigh11'] < df['dailyLow22']) & df['volfilt11'] & df['atrfilt11'] & df['locfilts11']
    
    # Entry conditions
    df['long_entry'] = df['bfvg11'] & (df['lastSwingType11'] == 'dailyLow') & df['in_trading_window']
    df['short_entry'] = df['sfvg11'] & (df['lastSwingType11'] == 'dailyHigh') & df['in_trading_window']
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i == 0:
            continue
        if pd.isna(df['dailyHigh11'].iloc[i]) or pd.isna(df['dailyHigh22'].iloc[i]):
            continue
        if pd.isna(df['dailyLow11'].iloc[i]) or pd.isna(df['dailyLow22'].iloc[i]):
            continue
            
        if df['long_entry'].iloc[i]:
            ts = int(df['time'].iloc[i])
            price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
        elif df['short_entry'].iloc[i]:
            ts = int(df['time'].iloc[i])
            price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
    
    return entries