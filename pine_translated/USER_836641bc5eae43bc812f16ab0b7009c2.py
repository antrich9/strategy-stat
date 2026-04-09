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
    
    # Ensure we have required columns
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Convert time to datetime for timezone-aware operations
    data['datetime'] = pd.to_datetime(data['time'], unit='s', utc=True)
    data['datetime'] = data['datetime'].dt.tz_convert('Europe/London')
    
    # Extract date components
    data['year'] = data['datetime'].dt.year
    data['month'] = data['datetime'].dt.month
    data['day'] = data['datetime'].dt.day
    data['hour'] = data['datetime'].dt.hour
    data['minute'] = data['datetime'].dt.minute
    
    # Define trading windows
    # Window 1: 07:00-11:45
    window1_start = (data['hour'] == 7) & (data['minute'] >= 0)
    window1_end = (data['hour'] == 11) & (data['minute'] <= 45)
    in_window1 = window1_start | window1_end
    
    # Actually, need to be more precise
    data['time_in_minutes'] = data['hour'] * 60 + data['minute']
    in_window1 = (data['time_in_minutes'] >= 7 * 60) & (data['time_in_minutes'] <= 11 * 60 + 45)
    in_window2 = (data['time_in_minutes'] >= 14 * 60) & (data['time_in_minutes'] <= 14 * 60 + 45)
    
    in_trading_window = in_window1 | in_window2
    
    # Wilder RSI implementation
    def wilder_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilder ATR implementation
    def wilder_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    # Calculate 4H data
    data['datetime_4h'] = data['datetime'].dt.floor('4H')
    grouped_4h = data.groupby('datetime_4h').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'open': 'first',
        'volume': 'sum'
    }).reset_index()
    
    # Create 4H shifted versions
    for col in ['high', 'low', 'close', 'open', 'volume']:
        grouped_4h[f'{col}_4h'] = grouped_4h[col].shift(1)
        grouped_4h[f'{col}_4h_2'] = grouped_4h[col].shift(2)
    
    # Merge 4H data back to main dataframe
    data = data.merge(grouped_4h[['datetime_4h', 'high_4h', 'low_4h', 'close_4h', 'open_4h', 'volume_4h',
                                   'high_4h_2', 'low_4h_2', 'close_4h_2']], 
                      left_on='datetime_4h', right_on='datetime_4h', how='left')
    
    # Fill NaN values with previous valid values (lookahead_off behavior)
    data['high_4h'] = data['high_4h'].ffill()
    data['low_4h'] = data['low_4h'].ffill()
    data['close_4h'] = data['close_4h'].ffill()
    data['open_4h'] = data['open_4h'].ffill()
    data['volume_4h'] = data['volume_4h'].ffill()
    data['high_4h_2'] = data['high_4h_2'].ffill()
    data['low_4h_2'] = data['low_4h_2'].ffill()
    data['close_4h_2'] = data['close_4h_2'].ffill()
    
    # Detect new 4H candle
    data['is_new_4h'] = data['datetime_4h'] != data['datetime_4h'].shift(1)
    
    # Volume Filter (4H)
    volume_sma_4h = data['volume_4h'].rolling(9).mean()
    volfilt1 = data['volume_4h'].shift(1) > volume_sma_4h.shift(1) * 1.5
    
    # ATR Filter (4H)
    atr_4h = wilder_atr(data['high_4h'], data['low_4h'], data['close_4h'], 20) / 1.5
    atrfilt1 = (data['low_4h'] - data['high_4h_2'] > atr_4h) | (data['low_4h_2'] - data['high_4h'] > atr_4h)
    
    # Trend Filter (4H)
    loc1 = data['close_4h'].rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    locfiltb1 = loc21
    locfilts1 = ~loc21
    
    # Bullish and Bearish FVGs (4H)
    bfvg1 = (data['low_4h'] > data['high_4h_2']) & volfilt1 & atrfilt1 & locfiltb1
    sfvg1 = (data['high_4h'] < data['low_4h_2']) & volfilt1 & atrfilt1 & locfilts1
    
    # Volume Filter (chart timeframe)
    volume_sma = data['volume'].rolling(9).mean()
    volfilt = data['volume'].shift(1) > volume_sma.shift(1) * 1.5
    
    # ATR Filter (chart timeframe)
    atr2 = wilder_atr(data['high'], data['low'], data['close'], 20) / 1.5
    atrfilt = (data['low'] - data['high'].shift(2) > atr2) | (data['low'].shift(2) - data['high'] > atr2)
    
    # Trend Filter (chart timeframe)
    loc = data['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # Bullish and Bearish FVGs (chart timeframe)
    bfvg = (data['low'] > data['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (data['high'] < data['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    # Track FVG state and entries
    entries = []
    trade_num = 1
    lastFVG = 0  # 0 = None, 1 = Bullish, -1 = Bearish
    
    # We need to iterate and detect sharp turns on confirmed new 4H candles
    data['prev_lastFVG'] = 0
    
    for i in range(1, len(data)):
        if pd.isna(data['high_4h'].iloc[i]) or pd.isna(data['low_4h'].iloc[i]):
            continue
            
        # Update lastFVG based on previous bar's conditions
        prev_lastFVG = lastFVG
        
        # Check for Sharp Turn (4H timeframe)
        if data['is_new_4h'].iloc[i] and in_trading_window.iloc[i]:
            # Bullish Sharp Turn: Bullish FVG after Bearish FVG
            if data['bfvg1'].iloc[i] and lastFVG == -1:
                entry_ts = int(data['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entry_price = float(data['close'].iloc[i])
                
                entries.append({
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
                trade_num += 1
                lastFVG = 1
                
            # Bearish Sharp Turn: Bearish FVG after Bullish FVG
            elif data['sfvg1'].iloc[i] and lastFVG == 1:
                entry_ts = int(data['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entry_price = float(data['close'].iloc[i])
                
                entries.append({
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
                trade_num += 1
                lastFVG = -1
                
            # Update FVG state if no sharp turn but FVG exists
            elif data['bfvg1'].iloc[i]:
                lastFVG = 1
            elif data['sfvg1'].iloc[i]:
                lastFVG = -1
    
    return entries