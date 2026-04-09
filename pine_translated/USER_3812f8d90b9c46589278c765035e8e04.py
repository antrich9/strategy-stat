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
    
    # Create 4H resampled data
    ohlc_4h = df.set_index('time').resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().copy()
    
    # Create daily resampled data
    ohlc_daily = df.set_index('time').resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().copy()
    
    # Extract hour and minute for time window (London timezone)
    df['hour'] = df['time'].dt.tz_convert('Europe/London').dt.hour
    df['minute'] = df['time'].dt.tz_convert('Europe/London').dt.minute
    
    def in_trading_window(row):
        t = row['hour'] * 60 + row['minute']
        return (7*60+45 <= t <= 11*60+45) or (14*60 <= t <= 14*60+45)
    
    df['in_trading_window'] = df.apply(in_trading_window, axis=1)
    
    # Daily swing detection
    # swing_high: dailyHigh21 < dailyHigh22 AND dailyHigh11[3] < dailyHigh22 AND dailyHigh11[4] < dailyHigh22
    ohlc_daily['is_swing_high'] = (
        (ohlc_daily['high'].shift(1) < ohlc_daily['high'].shift(2)) &
        (ohlc_daily['high'].shift(3) < ohlc_daily['high'].shift(2)) &
        (ohlc_daily['high'].shift(4) < ohlc_daily['high'].shift(2))
    )
    
    # swing_low: dailyLow21 > dailyLow22 AND dailyLow11[3] > dailyLow22 AND dailyLow11[4] > dailyLow22
    ohlc_daily['is_swing_low'] = (
        (ohlc_daily['low'].shift(1) > ohlc_daily['low'].shift(2)) &
        (ohlc_daily['low'].shift(3) > ohlc_daily['low'].shift(2)) &
        (ohlc_daily['low'].shift(4) > ohlc_daily['low'].shift(2))
    )
    
    # Track last swing type
    lastSwingType11 = 'none'
    swing_records = []
    for idx in ohlc_daily.index:
        if ohlc_daily.loc[idx, 'is_swing_high']:
            lastSwingType11 = 'dailyHigh'
        elif ohlc_daily.loc[idx, 'is_swing_low']:
            lastSwingType11 = 'dailyLow'
        swing_records.append({'date': idx.date(), 'lastSwingType11': lastSwingType11})
    
    swing_df = pd.DataFrame(swing_records)
    
    # Merge daily swing type to main df
    df['date'] = df['time'].dt.date
    swing_df['date'] = pd.to_datetime(swing_df['date'])
    df = df.merge(swing_df, on='date', how='left')
    
    # Align 4H data with 15m bars
    df['time_4h'] = df['time'].dt.floor('4h')
    ohlc_4h_reset = ohlc_4h.reset_index()
    ohlc_4h_reset.columns = ['time_4h', 'open_4h', 'high_4h', 'low_4h', 'close_4h', 'volume_4h']
    df = df.merge(ohlc_4h_reset, on='time_4h', how='left')
    
    # 4H FVG conditions
    # bullish FVG: low_4h > high_4h_2 (current 4H low > high 2 periods ago)
    df['bfvg'] = df['low_4h'] > df['high_4h'].shift(2)
    # bearish FVG: high_4h < low_4h_2 (current 4H high < low 2 periods ago)
    df['sfvg'] = df['high_4h'] < df['low_4h'].shift(2)
    
    # Skip last row to avoid repainting (barstate.isconfirmed = false for current bar)
    df = df.iloc[:-1]
    
    # Build entry conditions
    df['long_entry'] = df['bfvg'] & df['in_trading_window'] & (df['lastSwingType11'] == 'dailyLow')
    df['short_entry'] = df['sfvg'] & df['in_trading_window'] & (df['lastSwingType11'] == 'dailyHigh')
    
    # Generate entries
    results = []
    trade_num = 1
    
    for i, row in df.iterrows():
        if row['long_entry'] and not pd.isna(row['close']):
            ts = int(row['time'].timestamp())
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(row['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['close']),
                'raw_price_b': float(row['close'])
            })
            trade_num += 1
        elif row['short_entry'] and not pd.isna(row['close']):
            ts = int(row['time'].timestamp())
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(row['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['close']),
                'raw_price_b': float(row['close'])
            })
            trade_num += 1
    
    return results