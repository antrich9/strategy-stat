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
    
    entries = []
    trade_num = 1
    
    # ADX Parameters
    adxLength = 14
    adxThreshold = 25
    
    # Calculate DMI (ADX)
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # ATR calculation (Wilder)
    tr = pd.concat([df['high'] - df['low'], 
                   abs(df['high'] - df['close'].shift(1)), 
                   abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/adxLength, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/adxLength, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/adxLength, adjust=False).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/adxLength, adjust=False).mean()
    
    strong_trend = adx > adxThreshold
    
    # EMA calculation (5-min timeframe simulation using 5-period rolling)
    ema_fast = df['close'].ewm(span=10, adjust=False).mean()
    ema_slow = df['close'].ewm(span=50, adjust=False).mean()
    
    ema_bullish = ema_fast > ema_slow
    ema_bearish = ema_fast < ema_slow
    
    # FVG Detection (Bullish: gap up, Bearish: gap down)
    # Bullish FVG: low >= high[2] (current low is above high 2 bars ago)
    # Bearish FVG: high <= low[2] (current high is below low 2 bars ago)
    bullish_fvg = (df['low'] >= df['high'].shift(2))
    bearish_fvg = (df['high'] <= df['low'].shift(2))
    
    # FVG zone boundaries
    fvg_top_bull = df['high'].shift(2)  # Top of bullish FVG
    fvg_bottom_bull = df['low']  # Bottom of bullish FVG
    fvg_top_bear = df['high']  # Top of bearish FVG
    fvg_bottom_bear = df['low'].shift(2)  # Bottom of bearish FVG
    
    # Time window conditions (simplified - assume bars are at various times)
    # For backtesting, we check if trend aligns with time window logic
    is_bullish_time = ema_bullish
    is_bearish_time = ema_bearish
    
    # Entry conditions
    long_condition = (
        strong_trend & 
        ema_bullish & 
        bullish_fvg
    )
    
    short_condition = (
        strong_trend & 
        ema_bearish & 
        bearish_fvg
    )
    
    for i in range(len(df)):
        if pd.isna(adx.iloc[i]) or pd.isna(ema_fast.iloc[i]) or pd.isna(ema_slow.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        
        elif short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries