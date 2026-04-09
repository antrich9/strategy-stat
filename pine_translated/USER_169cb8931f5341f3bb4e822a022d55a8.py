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
    
    ema20Len = 20
    ema50Len = 50
    atrLen = 14
    wickPct = 40.0
    dojiPct = 5.0
    tradeDir = "Both"
    
    ema20 = df['close'].ewm(span=ema20Len, adjust=False).mean()
    ema50 = df['close'].ewm(span=ema50Len, adjust=False).mean()
    
    high = df['high']
    low = df['low']
    close = df['close']
    open_ = df['open']
    
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = true_range.ewm(alpha=1/atrLen, adjust=False).mean()
    
    zoneHigh = pd.concat([ema20, ema50], axis=1).max(axis=1)
    zoneLow = pd.concat([ema20, ema50], axis=1).min(axis=1)
    
    cRange = high - low
    bodyHi = pd.concat([open_, close], axis=1).max(axis=1)
    bodyLo = pd.concat([open_, close], axis=1).min(axis=1)
    bodySize = bodyHi - bodyLo
    lowerWick = bodyLo - low
    upperWick = high - bodyHi
    
    isDoji = (cRange > 0) & ((bodySize / cRange * 100) <= dojiPct)
    isLowTestWick = (cRange > 0) & ((lowerWick / cRange * 100) >= wickPct)
    isHighTestWick = (cRange > 0) & ((upperWick / cRange * 100) >= wickPct)
    
    low_diff = np.abs(low - low.shift(1))
    isTweezerBot = (low_diff <= atr * 0.1) & (close > open_) & (close.shift(1) <= open_.shift(1))
    
    high_diff = np.abs(high - high.shift(1))
    isTweezerTop = (high_diff <= atr * 0.1) & (close < open_) & (close.shift(1) >= open_.shift(1))
    
    touchZoneLong = (low <= zoneHigh) & (low >= zoneLow - atr * 0.5)
    touchZoneShort = (high >= zoneLow) & (high <= zoneHigh + atr * 0.5)
    
    bullTrend = ema20 > ema50
    bearTrend = ema20 < ema50
    
    bullPattern = isDoji | isLowTestWick | isTweezerBot
    bearPattern = isDoji | isHighTestWick | isTweezerTop
    
    longCondition = bullTrend & touchZoneLong & bullPattern & ((tradeDir == "Long") | (tradeDir == "Both"))
    shortCondition = bearTrend & touchZoneShort & bearPattern & ((tradeDir == "Short") | (tradeDir == "Both"))
    
    entries = []
    trade_num = 1
    
    valid_idx = df.index[~ema20.isna() & ~ema50.isna() & ~atr.isna()]
    
    for i in valid_idx:
        if longCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        elif shortCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    return entries