import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Volume filter: volume[1] > sma(volume, 9) * 1.5
    df['volfilt'] = df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5
    
    # Wilder ATR (ta.atr(20) / 1.5)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    
    # ATR filter: (low - high[2] > atr) or (low[2] - high > atr)
    atrfilt = (low - high.shift(2) > atr) | (low.shift(2) - high > atr)
    
    # Trend filter: sma(close, 54) > sma(close, 54)[1]
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # Bullish FVG: low > high[2] and volfilt and atrfilt and locfiltb
    bfvg = (low > high.shift(2)) & df['volfilt'] & atrfilt & locfiltb
    
    # Bearish FVG: high < low[2] and volfilt and atrfilt and locfilts
    sfvg = (high < low.shift(2)) & df['volfilt'] & atrfilt & locfilts
    
    # Extract hour and minute from timestamp
    dt = pd.to_datetime(df['time'], unit='s', utc=True)
    hours = dt.dt.hour
    minutes = dt.dt.minute
    
    # Trading window 1: 07:00 - 10:59
    in_window1 = ((hours > 7) | ((hours == 7) & (minutes >= 0))) & \
                 ((hours < 11) | ((hours == 10) & (minutes <= 59)))
    
    # Trading window 2: 15:00 - 16:59
    in_window2 = ((hours > 15) | ((hours == 15) & (minutes >= 0))) & \
                 ((hours < 17) | ((hours == 16) & (minutes <= 59)))
    
    in_trading_window = in_window1 | in_window2
    
    # Entry conditions
    long_cond = bfvg & in_trading_window
    short_cond = sfvg & in_trading_window
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        
        if short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries