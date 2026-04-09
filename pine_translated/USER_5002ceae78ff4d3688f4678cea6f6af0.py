import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.sort_values('time').reset_index(drop=True)
    
    # Aggregate to 4H timeframe from provided data
    df['4h_time'] = df['datetime'].dt.floor('4H')
    df_4h = df.groupby('4h_time').agg({
        'time': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index(drop=True)
    
    # Aggregate to daily timeframe
    df['date'] = df['datetime'].dt.date
    df_daily = df.groupby('date').agg({
        'time': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index(drop=True)
    
    # Calculate previous day high/low for sweep detection
    df_daily['prev_day_high'] = df_daily['high'].shift(1)
    df_daily['prev_day_low'] = df_daily['low'].shift(1)
    df_daily['new_day'] = True
    df_daily['new_day'].iloc[1:] = df_daily['date'].iloc[1:] != df_daily['date'].iloc[:-1].values
    
    # Sweep detection
    df['swept_high'] = False
    df['swept_low'] = False
    
    # Calculate swing highs/lows for 4H data
    swing_high = (df_4h['high'] > df_4h['high'].shift(1)) & (df_4h['high'] > df_4h['high'].shift(-1)) & \
                 (df_4h['high'].shift(1) > df_4h['high'].shift(2))
    swing_low = (df_4h['low'] < df_4h['low'].shift(1)) & (df_4h['low'] < df_4h['low'].shift(-1)) & \
                (df_4h['low'].shift(1) < df_4h['low'].shift(2))
    
    # FVG conditions using 4H data
    # Bullish FVG: current 4H low > high 2 bars ago
    # Bearish FVG: current 4H high < low 2 bars ago
    df_4h['bullish_fvg'] = df_4h['low'] > df_4h['high'].shift(2)
    df_4h['bearish_fvg'] = df_4h['high'] < df_4h['low'].shift(2)
    
    # Volume filter (4H SMA)
    df_4h['vol_sma'] = df_4h['volume'].rolling(9).mean()
    df_4h['volfilt'] = df_4h['volume'] > df_4h['vol_sma'] * 1.5
    
    # ATR filter (4H)
    tr_4h = pd.concat([df_4h['high'] - df_4h['low'], 
                       abs(df_4h['high'] - df_4h['close'].shift(1)),
                       abs(df_4h['low'] - df_4h['close'].shift(1))], axis=1).max(axis=1)
    atr_4h = tr_4h.ewm(alpha=1/20, adjust=False).mean() / 1.5
    
    # 4H trend filter (SMA 54)
    df_4h['trend_sma'] = df_4h['close'].rolling(54).mean()
    df_4h['trend_bullish'] = df_4h['trend_sma'] > df_4h['trend_sma'].shift(1)
    
    # Create London time windows
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['london_hour'] = df['datetime'].dt.tz_convert('Europe/London').dt.hour
    df['london_minute'] = df['datetime'].dt.tz_convert('Europe/London').dt.minute
    
    # Trading windows: 06:45-08:55 and 12:45-15:55 London time
    in_morning_window = ((df['london_hour'] == 6) & (df['london_minute'] >= 45)) | \
                        ((df['london_hour'] == 7) & (df['london_minute'] <= 55)) | \
                        ((df['london_hour'] == 8) & (df['london_minute'] <= 55))
    
    in_afternoon_window = ((df['london_hour'] == 12) & (df['london_minute'] >= 45)) | \
                          ((df['london_hour'] == 13) & (df['london_minute'] <= 55)) | \
                          ((df['london_hour'] == 14) & (df['london_minute'] <= 55)) | \
                          ((df['london_hour'] == 15) & (df['london_minute'] <= 55))
    
    df['in_trading_window'] = in_morning_window | in_afternoon_window
    
    # Detect new day
    df['prev_date'] = df['date'].shift(1)
    df['is_new_day'] = df['date'] != df['prev_date']
    df.loc[df['prev_date'].isna(), 'is_new_day'] = True
    
    # Reset sweep flags at start of each day
    df['swept_high_today'] = False
    df['swept_low_today'] = False
    
    # Build entry signals
    df['bullish_entry'] = False
    df['bearish_entry'] = False
    
    # Create a mapping from 4H time to FVG and filters
    df_4h_indexed = df_4h.set_index('4h_time')
    
    # Map 4H data back to main dataframe
    df['4h_bullish_fvg'] = df['4h_time'].map(df_4h_indexed['bullish_fvg']).fillna(False)
    df['4h_bearish_fvg'] = df['4h_time'].map(df_4h_indexed['bearish_fvg']).fillna(False)
    df['4h_volfilt'] = df['4h_time'].map(df_4h_indexed['volfilt']).fillna(True)
    df['4h_trend_bullish'] = df['4h_time'].map(df_4h_indexed['trend_bullish']).fillna(True)
    
    # Map daily data for sweep detection
    df_daily_indexed = df_daily.set_index('date')
    df['prev_day_high'] = df['date'].map(df_daily_indexed['prev_day_high'])
    df['prev_day_low'] = df['date'].map(df_daily_indexed['prev_day_low'])
    
    # Sweep detection: high crosses above prev day high, low crosses below prev day low
    df['sweep_high'] = (df['high'] > df['prev_day_high']) & (df['low'].shift(1) <= df['prev_day_high'].shift(1))
    df['sweep_low'] = (df['low'] < df['prev_day_low']) & (df['high'].shift(1) >= df['prev_day_low'].shift(1))
    
    # Build entries list
    entries = []
    trade_num = 1
    
    prev_swept_high = False
    prev_swept_low = False
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Skip if in trading window
        if not row['in_trading_window']:
            continue
        
        # Reset sweep flags on new day
        if row['is_new_day']:
            prev_swept_high = False
            prev_swept_low = False
        
        # Update sweep flags
        if row['sweep_high']:
            prev_swept_high = True
        if row['sweep_low']:
            prev_swept_low = True
        
        # Check if FVG is from recent 4H candle (within last 4 bars)
        current_4h_time = row['4h_time']
        recent_4h_mask = (df_4h['4h_time'] >= current_4h_time - pd.Timedelta(hours=12)) & \
                        (df_4h['4h_time'] <= current_4h_time)
        
        if recent_4h_mask.any():
            recent_fvg = df_4h[recent_4h_mask]
            recent_bullish_fvg = recent_fvg['bullish_fvg'].any()
            recent_bearish_fvg = recent_fvg['bearish_fvg'].any()
        else:
            recent_bullish_fvg = row['4h_bullish_fvg']
            recent_bearish_fvg = row['4h_bearish_fvg']
        
        # Bullish entry: bullish FVG + sweep low (sweep of prev day low) + in trading window + filters
        if recent_bullish_fvg and prev_swept_low and row['4h_volfilt'] and row['4h_trend_bullish']:
            entry_time = datetime.fromtimestamp(row['time'], tz=timezone.utc)
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(row['time']),
                'entry_time': entry_time.isoformat(),
                'entry_price_guess': float(row['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['close']),
                'raw_price_b': float(row['close'])
            })
            trade_num += 1
            prev_swept_low = False  # Reset after entry
        
        # Bearish entry: bearish FVG + sweep high (sweep of prev day high) + in trading window + filters
        elif recent_bearish_fvg and prev_swept_high and row['4h_volfilt'] and not row['4h_trend_bullish']:
            entry_time = datetime.fromtimestamp(row['time'], tz=timezone.utc)
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(row['time']),
                'entry_time': entry_time.isoformat(),
                'entry_price_guess': float(row['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['close']),
                'raw_price_b': float(row['close'])
            })
            trade_num += 1
            prev_swept_high = False  # Reset after entry
    
    return entries