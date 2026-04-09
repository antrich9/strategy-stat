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
    
    # Ensure we have required columns
    required = ['time', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
        return results
    
    # Create local copies to avoid SettingWithCopyWarning
    df = df.copy()
    df['ts'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['ts'].dt.date
    
    # Aggregate to 4H data using resampling
    df_4h = df.set_index('ts').resample('4h', offset='0min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    df_4h['time'] = df_4h['ts'].astype(np.int64) // 10**9
    
    # Get daily data
    df_daily = df.set_index('ts').resample('1d', offset='0min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    df_daily['time'] = df_daily['ts'].astype(np.int64) // 10**9
    
    # Create shifted versions for 4H data
    df_4h['high_1'] = df_4h['high'].shift(1)
    df_4h['low_1'] = df_4h['low'].shift(1)
    df_4h['close_1'] = df_4h['close'].shift(1)
    df_4h['high_2'] = df_4h['high'].shift(2)
    df_4h['low_2'] = df_4h['low'].shift(2)
    df_4h['close_2'] = df_4h['close'].shift(2)
    
    # Create shifted versions for daily data
    df_daily['high_1'] = df_daily['high'].shift(1)
    df_daily['low_1'] = df_daily['low'].shift(1)
    df_daily['high_2'] = df_daily['high'].shift(2)
    df_daily['low_2'] = df_daily['low'].shift(2)
    
    # Swing detection on daily: dailyHigh21 < dailyHigh22 and dailyHigh11[3] < dailyHigh22 and dailyHigh11[4] < dailyHigh22
    # dailyHigh21 = prevDayHigh11 = daily high shift(1)
    # dailyHigh22 = high shift(2)
    # dailyHigh11[3] = high shift(3)
    # dailyHigh11[4] = high shift(4)
    df_daily['is_swing_high'] = (
        (df_daily['high_1'] < df_daily['high_2']) &
        (df_daily['high'].shift(3) < df_daily['high_2']) &
        (df_daily['high'].shift(4) < df_daily['high_2'])
    )
    
    df_daily['is_swing_low'] = (
        (df_daily['low_1'] > df_daily['low_2']) &
        (df_daily['low'].shift(3) > df_daily['low_2']) &
        (df_daily['low'].shift(4) > df_daily['low_2'])
    )
    
    # Track last swing type
    df_daily['lastSwingType'] = "none"
    last_swing_high = np.nan
    last_swing_low = np.nan
    
    for idx in range(len(df_daily)):
        if df_daily['is_swing_high'].iloc[idx] if idx < len(df_daily) else False:
            last_swing_high = df_daily['high_2'].iloc[idx]
            last_swing_low = np.nan
        if df_daily['is_swing_low'].iloc[idx] if idx < len(df_daily) else False:
            last_swing_low = df_daily['low_2'].iloc[idx]
            last_swing_high = np.nan
        if not np.isnan(last_swing_high):
            df_daily.iloc[idx, df_daily.columns.get_loc('lastSwingType')] = "dailyHigh"
        elif not np.isnan(last_swing_low):
            df_daily.iloc[idx, df_daily.columns.get_loc('lastSwingType')] = "dailyLow"
    
    # Previous day high and low (pdHigh, pdLow)
    df_daily['pdHigh'] = df_daily['high_1']
    df_daily['pdLow'] = df_daily['low_1']
    
    # Sweep detection
    df['pdHigh'] = df['date'].map(df_daily.set_index(df_daily['ts'].dt.date)['pdHigh'])
    df['pdLow'] = df['date'].map(df_daily.set_index(df_daily['ts'].dt.date)['pdLow'])
    
    # Mark new day
    df['newDay'] = df['date'].diff().fillna(False).astype(bool)
    
    # Sweep conditions
    df['sweptHigh'] = df['high'] > df['pdHigh']
    df['sweptLow'] = df['low'] < df['pdLow']
    
    # Merge 4H data to main df based on time alignment
    df['ts_4h'] = df['ts'].dt.floor('4h')
    df_4h_merged = df_4h.copy()
    df_4h_merged['ts_4h'] = df_4h_merged['ts']
    
    df = df.merge(df_4h_merged[['ts_4h', 'high_1', 'low_1', 'close_1', 'high_2', 'low_2', 'close_2']], 
                 on='ts_4h', how='left', suffixes=('', '_4h'))
    
    # FVG conditions
    df['bfvg_condition'] = df['low_1'] > df['high_2']
    df['sfvg_condition'] = df['high_1'] < df['low_2']
    
    # London trading windows (hour is in UTC, need to convert to London time)
    # Since we don't have timezone conversion, we check hour >= 7 and < 12 for first window
    # and hour >= 14 and < 15 for second window
    df['hour'] = df['ts'].dt.hour
    london_start_window1 = 7
    london_end_window1 = 11
    london_start_window2 = 14
    london_end_window2 = 14
    
    df['isWithinWindow1'] = (df['hour'] >= london_start_window1) & (df['hour'] < london_end_window1 + 45/60)
    df['isWithinWindow2'] = (df['hour'] >= london_start_window2) & (df['hour'] < london_end_window2 + 45/60)
    df['in_trading_window'] = df['isWithinWindow1'] | df['isWithinWindow2']
    
    # Get lastSwingType for each bar by mapping from daily
    df['lastSwingType'] = df['date'].map(df_daily.set_index(df_daily['ts'].dt.date)['lastSwingType']).fillna("none")
    
    # Entry condition: bfvg_condition and lastSwingType11 == "dailyLow"
    df['entry_condition'] = df['bfvg_condition'] & (df['lastSwingType'] == "dailyLow")
    
    # Barstate.isconfirmed - on 15min timeframe this is essentially all bars, so we accept all bars
    # Generate entries
    for i in range(len(df)):
        if df['entry_condition'].iloc[i]:
            entry_price = df['close'].iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            results.append({
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
    
    return results