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
    
    # Settings (default values from Pine Script inputs)
    ext11 = 10
    inp1_11 = False  # Volume Filter
    inp2_11 = False  # ATR Filter
    inp3_11 = False  # Trend Filter
    atrMultiplier11 = 3.0
    
    # Convert time to datetime for window filtering
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Calculate weekly aggregations for multi-timeframe analysis
    # Since we don't have weekly data, we'll simulate it
    df['week'] = df['datetime'].dt.to_period('W')
    weekly_agg = df.groupby('week').agg({
        'high': 'max',
        'low': 'min',
        'open': 'first',
        'close': 'last',
        'volume': 'sum',
        'time': 'last'
    }).reset_index(drop=True)
    
    # Create shifted versions for previous weeks
    for col in ['high', 'low', 'open', 'close']:
        weekly_agg[f'{col}21'] = weekly_agg[col].shift(1)
        weekly_agg[f'{col}22'] = weekly_agg[col].shift(2)
    
    # Current week data
    weekly_agg['dailyHigh11'] = weekly_agg['high']
    weekly_agg['dailyLow11'] = weekly_agg['low']
    weekly_agg['dailyClose11'] = weekly_agg['close']
    weekly_agg['dailyOpen11'] = weekly_agg['open']
    
    # Previous week data
    weekly_agg['prevDayHigh11'] = weekly_agg['high21']
    weekly_agg['prevDayLow11'] = weekly_agg['low21']
    weekly_agg['dailyHigh21'] = weekly_agg['high21']
    weekly_agg['dailyLow21'] = weekly_agg['low21']
    weekly_agg['dailyHigh22'] = weekly_agg['high22']
    weekly_agg['dailyLow22'] = weekly_agg['low22']
    
    # Calculate ATR (Wilder RSI method)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Wilder ATR
    atr1 = true_range.ewm(alpha=1/14, adjust=False).mean()
    atr211 = atr1 / 1.5
    
    # Volume filter
    sma_vol = df['volume'].rolling(9).mean()
    volfilt11 = (df['volume'].shift(1) > sma_vol.shift(1) * 1.5) if inp1_11 else True
    
    # ATR filter
    if inp2_11:
        atrfilt11 = (weekly_agg['dailyLow22'] - weekly_agg['dailyHigh22'] > atr211.values) | (weekly_agg['dailyLow22'] - weekly_agg['dailyHigh11'] > atr211.values)
    else:
        atrfilt11 = pd.Series([True] * len(df))
    
    # Trend filter
    loc11 = df['close'].rolling(54).mean()
    loc211 = loc11 > loc11.shift(1)
    locfiltb11 = loc211 if inp3_11 else pd.Series([True] * len(df))
    locfilts11 = ~loc211 if inp3_11 else pd.Series([True] * len(df))
    
    # Merge weekly indicators back to daily dataframe
    df['dailyHigh11'] = weekly_agg['dailyHigh11'].values[df['week'].map({v: k for k, v in enumerate(weekly_agg['week'].values)})]
    df['dailyLow11'] = weekly_agg['dailyLow11'].values[df['week'].map({v: k for k, v in enumerate(weekly_agg['week'].values)})]
    df['dailyHigh22'] = weekly_agg['dailyHigh22'].values[df['week'].map({v: k for k, v in enumerate(weekly_agg['week'].values)})]
    df['dailyLow22'] = weekly_agg['dailyLow22'].values[df['week'].map({v: k for k, v in enumerate(weekly_agg['week'].values)})]
    df['dailyHigh21'] = weekly_agg['dailyHigh21'].values[df['week'].map({v: k for k, v in enumerate(weekly_agg['week'].values)})]
    df['dailyLow21'] = weekly_agg['dailyLow21'].values[df['week'].map({v: k for k, v in enumerate(weekly_agg['week'].values)})]
    df['prevDayHigh11'] = weekly_agg['prevDayHigh11'].values[df['week'].map({v: k for k, v in enumerate(weekly_agg['week'].values)})]
    df['prevDayLow11'] = weekly_agg['prevDayLow11'].values[df['week'].map({v: k for k, v in enumerate(weekly_agg['week'].values)})]
    df['volfilt11'] = volfilt11
    df['atrfilt11'] = atrfilt11
    df['locfiltb11'] = locfiltb11
    df['locfilts11'] = locfilts11
    
    # Swing detection
    # is_swing_high11 = dailyHigh21 < dailyHigh22 and dailyHigh11[3] < dailyHigh22 and dailyHigh11[4] < dailyHigh22
    # is_swing_low11 = dailyLow21 > dailyLow22 and dailyLow11[3] > dailyLow22 and dailyLow11[4] > dailyLow22
    df['is_swing_high11'] = (df['dailyHigh21'] < df['dailyHigh22']) & (df['dailyHigh11'].shift(3) < df['dailyHigh22']) & (df['dailyHigh11'].shift(4) < df['dailyHigh22'])
    df['is_swing_low11'] = (df['dailyLow21'] > df['dailyLow22']) & (df['dailyLow11'].shift(3) > df['dailyLow22']) & (df['dailyLow11'].shift(4) > df['dailyLow22'])
    
    # Track last swing type
    last_swing_high11 = np.nan
    last_swing_low11 = np.nan
    lastSwingType11 = "none"
    
    # FVG detection
    bfvg11 = (df['dailyLow11'] > df['dailyHigh22']) & df['volfilt11'] & df['atrfilt11'] & df['locfiltb11']
    sfvg11 = (df['dailyHigh11'] < df['dailyLow22']) & df['volfilt11'] & df['atrfilt11'] & df['locfilts11']
    
    # Trading windows (London time)
    # Convert timestamps to London time
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    
    # Window 1: 07:00-09:45 London
    window1_start = (df['hour'] == 7) & (df['minute'] >= 0)
    window1_end = (df['hour'] == 9) & (df['minute'] <= 45)
    isWithinWindow1 = window1_start | window1_end
    
    # Window 2: 14:00-16:45 London
    window2_start = (df['hour'] == 14) & (df['minute'] >= 0)
    window2_end = (df['hour'] == 16) & (df['minute'] <= 45)
    isWithinWindow2 = window2_start | window2_end
    
    in_trading_window = isWithinWindow1 | isWithinWindow2
    
    # Generate entries
    entries = []
    trade_num = 1
    
    # Need to iterate and track state
    for i in range(len(df)):
        # Update swing tracking
        if df['is_swing_high11'].iloc[i] if pd.notna(df['is_swing_high11'].iloc[i]) else False:
            last_swing_high11 = df['dailyHigh22'].iloc[i]
            lastSwingType11 = "dailyHigh"
        if df['is_swing_low11'].iloc[i] if pd.notna(df['is_swing_low11'].iloc[i]) else False:
            last_swing_low11 = df['dailyLow22'].iloc[i]
            lastSwingType11 = "dailyLow"
        
        # Check entry conditions
        if i > 0 and in_trading_window.iloc[i]:
            # Bullish entry
            bfvg_val = bfvg11.iloc[i] if pd.notna(bfvg11.iloc[i]) else False
            is_swing_low = lastSwingType11 == "dailyLow"
            
            if bfvg_val and is_swing_low:
                entry_price = df['close'].iloc[i]
                entry_ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': float(entry_price),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(entry_price),
                    'raw_price_b': float(entry_price)
                })
                trade_num += 1
            
            # Bearish entry
            sfvg_val = sfvg11.iloc[i] if pd.notna(sfvg11.iloc[i]) else False
            is_swing_high = lastSwingType11 == "dailyHigh"
            
            if sfvg_val and is_swing_high:
                entry_price = df['close'].iloc[i]
                entry_ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': float(entry_price),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(entry_price),
                    'raw_price_b': float(entry_price)
                })
                trade_num += 1
    
    return entries