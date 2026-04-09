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
    trade_num = 0
    
    # === Calculate EMAs (8, 20, 50) ===
    ema1_len = 8
    ema2_len = 20
    ema3_len = 50
    
    ema1 = df['close'].ewm(span=ema1_len, adjust=False).mean()
    ema2 = df['close'].ewm(span=ema2_len, adjust=False).mean()
    ema3 = df['close'].ewm(span=ema3_len, adjust=False).mean()
    
    # Bullish EMA: ema1 > ema2 > ema3
    bullish_ema = (ema1 > ema2) & (ema2 > ema3)
    # Bearish EMA: ema1 < ema2 < ema3
    bearish_ema = (ema1 < ema2) & (ema2 < ema3)
    
    # === Calculate ATR (20) using Wilder's method ===
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    atr = pd.Series(index=df.index, dtype=float)
    atr.iloc[0] = tr.iloc[0]
    for i in range(1, len(df)):
        atr.iloc[i] = (atr.iloc[i-1] * 19 + tr.iloc[i]) / 20
    
    # === Calculate SMA for volume filter ===
    vol_sma = df['volume'].rolling(9).mean()
    
    # === Calculate location filter (SMA 54) ===
    loc1 = df['close'].rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    
    # === Calculate daily high/low and swing detection ===
    # Convert time to datetime for grouping by day
    df['datetime'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df['day'] = df['datetime'].dt.date
    
    # Get previous day high/low
    daily_high = df.groupby('day')['high'].cummax().shift(1)
    daily_low = df.groupby('day')['low'].cummin().shift(1)
    
    # Detect sweeps (price crossing previous day high/low)
    swept_high = df['high'] > daily_high
    swept_low = df['low'] < daily_low
    
    # Initialize sweep tracking per day
    df['new_day'] = df['day'].shift(1).fillna(df['day']).ne(df['day'])
    swept_high_once = pd.Series(False, index=df.index)
    swept_low_once = pd.Series(False, index=df.index)
    
    last_swept_high = False
    last_swept_low = False
    
    for i in range(len(df)):
        if df['new_day'].iloc[i]:
            last_swept_high = False
            last_swept_low = False
        
        if not last_swept_high and df['high'].iloc[i] > daily_high.iloc[i]:
            swept_high_once.iloc[i] = True
            last_swept_high = True
        
        if not last_swept_low and df['low'].iloc[i] < daily_low.iloc[i]:
            swept_low_once.iloc[i] = True
            last_swept_low = True
    
    # === Calculate 4H FVG (Fair Value Gap) ===
    # Since we don't have 4H data directly, we'll approximate using local high/low comparisons
    # Lookback windows
    high_2 = df['high'].shift(2)
    low_2 = df['low'].shift(2)
    close_1 = df['close'].shift(1)
    
    # Bullish FVG: current low > high 2 bars ago
    bfvg = df['low'] > high_2
    # Bearish FVG: current high < low 2 bars ago
    sfvg = df['high'] < low_2
    
    # === Trading Window (London time 07:45-11:45 and 14:00-14:45) ===
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    
    # Window 1: 07:45 to 11:45
    window1_start = (df['hour'] == 7) & (df['minute'] >= 45)
    window1_mid = (df['hour'] >= 8) & (df['hour'] <= 10)
    window1_end = (df['hour'] == 11) & (df['minute'] <= 45)
    in_window1 = window1_start | window1_mid | window1_end
    
    # Window 2: 14:00 to 14:45
    window2_start = (df['hour'] == 14) & (df['minute'] >= 0) & (df['minute'] <= 45)
    in_window2 = window2_start
    
    in_trading_window = in_window1 | in_window2
    
    # === Apply Filters ===
    # Volume filter (disabled by default inp11 = false)
    volfilt = True  # inp11 = false
    # ATR filter (disabled by default inp21 = false)
    atrfilt = True  # inp21 = false
    # Trend filter (disabled by default inp31 = false)
    locfilt_long = True  # inp31 = false
    locfilt_short = True  # inp31 = false
    
    # === Combine Entry Conditions ===
    # Long: bullish EMA + bullish FVG + swept high + in window + filters
    long_condition = (
        bullish_ema & 
        bfvg & 
        swept_high_once & 
        in_trading_window & 
        volfilt & 
        atrfilt & 
        locfilt_long
    )
    
    # Short: bearish EMA + bearish FVG + swept low + in window + filters
    short_condition = (
        bearish_ema & 
        sfvg & 
        swept_low_once & 
        in_trading_window & 
        volfilt & 
        atrfilt & 
        locfilt_short
    )
    
    # === Generate Entries ===
    for i in range(1, len(df)):
        # Skip if any required indicator is NaN
        if pd.isna(ema1.iloc[i]) or pd.isna(ema2.iloc[i]) or pd.isna(ema3.iloc[i]):
            continue
        
        entry_price = df['close'].iloc[i]
        ts = df['time'].iloc[i]
        entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
        
        # Check long condition
        if long_condition.iloc[i]:
            trade_num += 1
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
        
        # Check short condition
        if short_condition.iloc[i]:
            trade_num += 1
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
    
    return results