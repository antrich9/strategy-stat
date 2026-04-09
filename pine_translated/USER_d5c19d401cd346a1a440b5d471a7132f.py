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
    # Constants from Pine Script
    useAllHours = False
    useLondon = True
    useNYAM = True
    useNYPM = False
    volMAPeriod = 20
    volMultiplier = 1.8  # Default for futures (isFutures=True)
    requireVolSurge = True
    atrPeriod = 14
    maxTradesPerDay = 3
    useTrendFilter = True
    
    results = []
    trade_num = 0
    trades_today = 0
    prev_date = None
    
    # Calculate EMAs
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    ema200 = df['close'].ewm(span=200, adjust=False).mean()
    
    # Calculate MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    
    # Calculate Volume SMA
    vol_ma = df['volume'].rolling(window=volMAPeriod).mean()
    
    # Calculate ATR (Wilder's smoothing)
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder's ATR
    atr = pd.Series(np.nan, index=df.index)
    first_atr = true_range.iloc[:atrPeriod].mean()
    atr.iloc[atrPeriod - 1] = first_atr
    multiplier = (atrPeriod - 1) / atrPeriod
    for i in range(atrPeriod, len(df)):
        atr.iloc[i] = atr.iloc[i - 1] * multiplier + true_range.iloc[i] / atrPeriod
    
    # Extract hour from timestamp
    hours = pd.to_datetime(df['time'], unit='s', utc=True).dt.hour
    
    # Build conditions
    uptrend = (ema50 > ema200) & (close > ema50)
    downtrend = (ema50 < ema200) & (close < ema50)
    trend_valid = (~useTrendFilter) | (uptrend | downtrend)
    
    macd_bullish = macd_line > macd_signal
    macd_bearish = macd_line < macd_signal
    
    volume_surge = df['volume'] > (vol_ma * volMultiplier)
    vol_confirmed = (~requireVolSurge) | volume_surge
    
    bullish_bar = (close - low) > (high - close)
    bearish_bar = (high - close) > (close - low)
    
    in_london = (hours >= 7) & (hours < 10)
    in_nyam = (hours >= 13) & (hours < 16)
    in_nypm = (hours >= 19) & (hours < 21)
    in_time_window = useAllHours | in_london | in_nyam | in_nypm
    
    long_condition = (uptrend & vol_confirmed & trend_valid & macd_bullish & 
                       bullish_bar & (close > ema50) & in_time_window)
    short_condition = (downtrend & vol_confirmed & trend_valid & 
                        macd_bearish & bearish_bar & (close < ema50) & in_time_window)
    
    # Iterate through bars
    in_position = False
    
    for i in range(len(df)):
        # Check for NaN in key indicators
        if pd.isna(ema50.iloc[i]) or pd.isna(ema200.iloc[i]) or pd.isna(macd_line.iloc[i]):
            continue
        if pd.isna(vol_ma.iloc[i]) or pd.isna(atr.iloc[i]):
            continue
            
        # Track new day and reset trade count
        current_ts = df['time'].iloc[i]
        current_dt = datetime.fromtimestamp(current_ts, tz=timezone.utc)
        current_date = current_dt.date()
        
        if prev_date is not None and current_date != prev_date:
            trades_today = 0
        prev_date = current_date
        
        # Skip if in position
        if in_position:
            continue
            
        # Check entry conditions
        if trades_today < maxTradesPerDay:
            if long_condition.iloc[i]:
                trade_num += 1
                trades_today += 1
                in_position = True
                
                entry_ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entry_price = float(df['close'].iloc[i])
                
                results.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                
            elif short_condition.iloc[i]:
                trade_num += 1
                trades_today += 1
                in_position = True
                
                entry_ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entry_price = float(df['close'].iloc[i])
                
                results.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
    
    return results