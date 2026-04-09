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
    results = []
    trade_num = 1
    
    # Input parameters (matching Pine Script defaults)
    inp1 = False  # Volume Filter
    inp2 = False  # ATR Filter
    inp3 = False  # Trend Filter
    atr_length_4h = 20
    vol_sma_len = 9
    trend_sma_len = 54
    atr_length = 14
    
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    
    # Trading windows (London time: UTC in winter, UTC+1 in summer)
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['year'] = df['datetime'].dt.year
    
    # Check if DST is in effect (simplified: April-October is BST)
    df['is_bst'] = df['month'].isin([4, 5, 6, 7, 8, 9, 10])
    df['london_offset'] = df['is_bst'].apply(lambda x: 1 if x else 0)
    df['london_hour'] = df['hour'] - df['london_offset']
    
    # Window 1: 07:00-11:45
    window1_start = (df['london_hour'] > 6) | ((df['london_hour'] == 6) & (df['minute'] >= 0))
    window1_end = (df['london_hour'] < 11) | ((df['london_hour'] == 11) & (df['minute'] < 45))
    is_within_window1 = window1_start & window1_end
    
    # Window 2: 14:00-14:45
    window2_start = (df['london_hour'] > 13) | ((df['london_hour'] == 13) & (df['minute'] >= 59))
    window2_end = (df['london_hour'] < 14) | ((df['london_hour'] == 14) & (df['minute'] < 45))
    is_within_window2 = window2_start & window2_end
    
    in_trading_window = is_within_window1 | is_within_window2
    
    # Resample to 4H candles
    df_4h = df.set_index('datetime').resample('240min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    df_4h['time'] = df_4h.index.astype('int64') // 10**9
    
    # Check for new 4H candle
    df_4h['is_new_4h'] = True
    
    # Volume Filter for 4H
    vol_sma_4h = df_4h['volume'].rolling(vol_sma_len).mean()
    volfilt1 = (df_4h['volume'].shift(1) > vol_sma_4h * 1.5) if inp1 else pd.Series(True, index=df_4h.index)
    
    # ATR Filter for 4H (Wilder ATR)
    high_arr = df_4h['high'].values
    low_arr = df_4h['low'].values
    close_arr = df_4h['close'].values
    
    tr_4h = np.maximum(high_arr[1:] - low_arr[1:], 
                       np.maximum(np.abs(high_arr[1:] - close_arr[:-1]),
                                  np.abs(low_arr[1:] - close_arr[:-1])))
    atr_4h = np.zeros(len(df_4h))
    atr_4h[atr_length_4h] = np.mean(tr_4h[:atr_length_4h])
    for i in range(atr_length_4h + 1, len(df_4h)):
        atr_4h[i] = (atr_4h[i-1] * (atr_length_4h - 1) + tr_4h[i-1]) / atr_length_4h
    atr_4h_series = pd.Series(atr_4h, index=df_4h.index)
    
    if inp2:
        gap_up = (df_4h['low'] - df_4h['high'].shift(2) > atr_4h_series / 1.5)
        gap_down = (df_4h['low'].shift(2) - df_4h['high'] > atr_4h_series / 1.5)
        atrfilt1 = gap_up | gap_down
    else:
        atrfilt1 = pd.Series(True, index=df_4h.index)
    
    # Trend Filter for 4H
    loc1 = df_4h['close'].rolling(trend_sma_len).mean()
    loc21 = loc1 > loc1.shift(1)
    locfiltb1 = loc21 if inp3 else pd.Series(True, index=df_4h.index)
    locfilts1 = ~loc21 if inp3 else pd.Series(True, index=df_4h.index)
    
    # FVG detection on 4H
    bfvg1 = (df_4h['low'] > df_4h['high'].shift(2)) & volfilt1 & atrfilt1 & locfiltb1
    sfvg1 = (df_4h['high'] < df_4h['low'].shift(2)) & volfilt1 & atrfilt1 & locfilts1
    
    # Track last FVG type
    last_fvg = 0
    entries = []
    
    for i in range(1, len(df_4h)):
        if pd.isna(bfvg1.iloc[i]) or pd.isna(sfvg1.iloc[i]):
            continue
        
        curr_time = df_4h['time'].iloc[i]
        dt = datetime.fromtimestamp(curr_time, tz=timezone.utc)
        
        # Check if we're in a new 4H candle (simulate barstate.isconfirmed logic)
        is_confirmed_bar = True
        
        if is_confirmed_bar and df_4h['is_new_4h'].iloc[i]:
            # Detect Sharp Turn in FVGs
            if bfvg1.iloc[i] and last_fvg == -1:
                entries.append({
                    'direction': 'long',
                    'entry_ts': curr_time,
                    'entry_time': dt.isoformat()
                })
                last_fvg = 1
            elif sfvg1.iloc[i] and last_fvg == 1:
                entries.append({
                    'direction': 'short',
                    'entry_ts': curr_time,
                    'entry_time': dt.isoformat()
                })
                last_fvg = -1
            elif bfvg1.iloc[i]:
                last_fvg = 1
            elif sfvg1.iloc[i]:
                last_fvg = -1
    
    # Map 4H entries back to 15min entries and filter by trading window
    df['datetime_idx'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    for entry in entries:
        entry_time = pd.to_datetime(entry['entry_time'])
        closest_idx = (df['datetime_idx'] - entry_time).abs().idxmin()
        
        if not in_trading_window.iloc[closest_idx]:
            continue
        
        entry_price = df['close'].iloc[closest_idx]
        entry_ts = df['time'].iloc[closest_idx]
        
        results.append({
            'trade_num': trade_num,
            'direction': entry['direction'],
            'entry_ts': entry_ts,
            'entry_time': entry['entry_time'],
            'entry_price_guess': entry_price,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': entry_price,
            'raw_price_b': entry_price
        })
        trade_num += 1
    
    return results