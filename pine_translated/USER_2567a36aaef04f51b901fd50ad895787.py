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
    
    # Input parameters (matching Pine Script defaults)
    inp1 = False  # Volume Filter
    inp2 = False  # ATR Filter
    inp3 = False  # Trend Filter
    atr_length1 = 20
    
    # Resample to 4H timeframe
    df['time_dt'] = pd.to_datetime(df['time'], unit='ms')
    df_4h = df.set_index('time_dt').resample('4h').agg({
        'time': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    df_4h['time'] = df_4h['time'].astype(int)
    
    # Calculate 4H indicators
    vol_sma_4h = df_4h['volume'].rolling(9).mean()
    atr_4h = calculate_wilder_atr(df_4h, atr_length1) / 1.5
    loc1 = df_4h['close'].rolling(54).mean()
    loc21 = loc1 > loc1.shift(1)
    
    # Volume Filter
    volfilt1 = ~inp1 | (df_4h['volume'].shift(1) > vol_sma_4h * 1.5)
    
    # ATR Filter
    atrfilt1 = ~inp2 | ((df_4h['low'] - df_4h['high'].shift(2) > atr_4h) | (df_4h['low'].shift(2) - df_4h['high'] > atr_4h))
    
    # Trend Filter
    locfiltb1 = ~inp3 | loc21
    locfilts1 = ~inp3 | ~loc21
    
    # Identify FVGs on 4H
    bfvg1 = (df_4h['low'] > df_4h['high'].shift(2)) & volfilt1 & atrfilt1 & locfiltb1
    sfvg1 = (df_4h['high'] < df_4h['low'].shift(2)) & volfilt1 & atrfilt1 & locfilts1
    
    # Detect new 4H candle
    is_new_4h = df_4h.index.to_series().diff().dt.total_seconds() > 14400
    
    # Track last FVG state (only update on new 4H candles)
    lastFVG = pd.Series(0, index=df_4h.index)
    
    for i in range(2, len(df_4h)):
        if is_new_4h.iloc[i]:
            if bfvg1.iloc[i-1]:
                lastFVG.iloc[i] = 1
            elif sfvg1.iloc[i-1]:
                lastFVG.iloc[i] = -1
            else:
                lastFVG.iloc[i] = lastFVG.iloc[i-1]
        else:
            lastFVG.iloc[i] = lastFVG.iloc[i-1]
    
    # Convert timestamps to America/New_York for window checking
    def to_ny_time(ts):
        return pd.Timestamp(ts, tz='America/New_York')
    
    ny_times = df['time_dt'].dt.tz_convert('America/New_York')
    hour_min = ny_times.dt.hour * 60 + ny_times.dt.minute
    
    # Window 1: 02:45-05:45 NY (07:00-07:45 London equivalent shifted)
    in_window1 = (hour_min >= 165) & (hour_min < 345)
    # Window 2: 08:45-10:45 NY
    in_window2 = (hour_min >= 525) & (hour_min < 645)
    # Window 3: London 14:00-15:45 = NY 09:00-10:45
    in_window3 = (hour_min >= 540) & (hour_min < 645)
    
    in_trading_window = in_window1 | in_window2 | in_window3
    
    # Process entries only on new 4H candles
    trade_active = False
    
    for i in range(2, len(df_4h)):
        if not is_new_4h.iloc[i]:
            continue
        
        ny_idx = to_ny_time(df_4h.index[i])
        h = ny_idx.hour * 60 + ny_idx.minute
        in_win = (h >= 165 and h < 345) or (h >= 525 and h < 645) or (h >= 540 and h < 645)
        
        if not in_win:
            continue
        
        prev_fvg = lastFVG.iloc[i-1] if i > 0 else 0
        
        # Bullish Sharp Turn: Bullish FVG after Bearish FVG
        if bfvg1.iloc[i] and prev_fvg == -1:
            trade_num += 1
            entry_ts = int(df_4h['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df_4h['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df_4h['close'].iloc[i]),
                'raw_price_b': float(df_4h['close'].iloc[i])
            })
        
        # Bearish Sharp Turn: Bearish FVG after Bullish FVG
        elif sfvg1.iloc[i] and prev_fvg == 1:
            trade_num += 1
            entry_ts = int(df_4h['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df_4h['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df_4h['close'].iloc[i]),
                'raw_price_b': float(df_4h['close'].iloc[i])
            })
    
    return results


def calculate_wilder_atr(df: pd.DataFrame, length: int) -> pd.Series:
    """Calculate Wilder's ATR"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr