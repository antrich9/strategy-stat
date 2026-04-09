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
    
    # Wilder RSI implementation
    def wilder_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilder ATR implementation
    def wilder_atr(high, low, close, period):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/period, adjust=False).mean()
        return atr
    
    # Resample to 4H
    df['ts'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df_4h = df.set_index('ts').resample('240T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    df_4h = df_4h.reset_index()
    
    # Get 4H bar times aligned to NY timezone
    df_4h['time_4h'] = df_4h['ts'].dt.tz_convert('America/New_York')
    df_4h['hour_4h'] = df_4h['time_4h'].dt.hour
    df_4h['minute_4h'] = df_4h['time_4h'].dt.minute
    
    # For each 15min bar, find corresponding 4H bar
    df['time_15m'] = df['ts'].dt.tz_convert('America/New_York')
    df['hour_15m'] = df['time_15m'].dt.hour
    df['minute_15m'] = df['time_15m'].dt.minute
    
    # Time windows (NY time): 02:45-05:45 and 08:45-10:45
    in_window1 = (df['hour_15m'] == 2) & (df['minute_15m'] >= 45)
    in_window1 |= (df['hour_15m'] == 3) | ((df['hour_15m'] == 4) & (df['minute_15m'] <= 45))
    in_window1 |= (df['hour_15m'] == 5) & (df['minute_15m'] <= 45)
    
    in_window2 = (df['hour_15m'] == 8) & (df['minute_15m'] >= 45)
    in_window2 |= (df['hour_15m'] == 9)
    in_window2 |= (df['hour_15m'] == 10) & (df['minute_15m'] <= 45)
    
    in_trading_window = in_window1 | in_window2
    
    # 4H indicators
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    close_4h = df_4h['close']
    volume_4h = df_4h['volume']
    
    volfilt1 = volume_4h.shift(1) > volume_4h.rolling(9).mean() * 1.5
    
    atr_4h1 = wilder_atr(high_4h, low_4h, close_4h, 20) / 1.5
    atrfilt1 = (low_4h - high_4h.shift(2) > atr_4h1) | (low_4h.shift(2) - high_4h > atr_4h1)
    
    loc1 = close_4h.rolling(54).mean()
    locfiltb1 = loc1 > loc1.shift(1)
    locfilts1 = ~locfiltb1
    
    bfvg1 = (low_4h > high_4h.shift(2)) & volfilt1 & atrfilt1 & locfiltb1
    sfvg1 = (high_4h < low_4h.shift(2)) & volfilt1 & atrfilt1 & locfilts1
    
    # Create 4H FVG series aligned with 15min bars
    df['bfvg1_4h'] = bfvg1.reindex(df_4h.index).values
    df['sfvg1_4h'] = sfvg1.reindex(df_4h.index).values
    
    entries = []
    trade_num = 1
    lastFVG = 0
    
    for i in range(len(df)):
        if not in_trading_window.iloc[i]:
            continue
        
        current_bfvg = df['bfvg1_4h'].iloc[i] if not pd.isna(df['bfvg1_4h'].iloc[i]) else False
        current_sfvg = df['sfvg1_4h'].iloc[i] if not pd.isna(df['sfvg1_4h'].iloc[i]) else False
        
        if current_bfvg and lastFVG == -1:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            lastFVG = 1
        elif current_sfvg and lastFVG == 1:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            lastFVG = -1
        elif current_bfvg:
            lastFVG = 1
        elif current_sfvg:
            lastFVG = -1
    
    return entries