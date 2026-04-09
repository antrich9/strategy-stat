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
    
    df = df.copy().reset_index(drop=True)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('time')
    
    # Resample to daily for daily data
    daily = df.resample('D').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'open': 'first'
    }).dropna()
    
    # Daily data references
    dailyHigh11 = daily['high']
    dailyLow11 = daily['low']
    dailyClose11 = daily['close']
    dailyOpen11 = daily['open']
    
    # Previous day data (shift by 1)
    prevDayHigh11 = daily['high'].shift(1)
    prevDayLow11 = daily['low'].shift(1)
    
    # Daily data with historical reference
    dailyHigh21 = daily['high'].shift(1)
    dailyLow21 = daily['low'].shift(1)
    dailyHigh22 = daily['high'].shift(2)
    dailyLow22 = daily['low'].shift(2)
    
    # Wilder ATR implementation
    def wilder_atr(df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    # Filter values (default disabled)
    volfilt = True
    atr211 = wilder_atr(df, 20) / 1.5
    atrfilt = True
    
    # Trend filter
    loc11 = df['close'].rolling(54).mean()
    loc211 = loc11 > loc11.shift(1)
    locfiltb = loc211
    locfilts = ~loc211
    
    # FVG detection
    bfvg = (dailyLow11 > dailyHigh22) & volfilt & atrfilt & locfiltb
    sfvg = (dailyHigh11 < dailyLow22) & volfilt & atrfilt & locfilts
    
    # Swing detection
    is_swing_high = (dailyHigh21 < dailyHigh22) & (dailyHigh11.shift(3) < dailyHigh22) & (dailyHigh11.shift(4) < dailyHigh22)
    is_swing_low = (dailyLow21 > dailyLow22) & (dailyLow11.shift(3) > dailyLow22) & (dailyLow11.shift(4) > dailyLow22)
    
    # Track last swing
    last_swing_high = pd.Series(index=daily.index, dtype=float)
    last_swing_low = pd.Series(index=daily.index, dtype=float)
    lastSwingType = pd.Series(index=daily.index, dtype=str)
    
    for i in range(len(daily)):
        if i > 0:
            last_swing_high.iloc[i] = last_swing_high.iloc[i-1]
            last_swing_low.iloc[i] = last_swing_low.iloc[i-1]
            lastSwingType.iloc[i] = lastSwingType.iloc[i-1]
        
        if is_swing_high.iloc[i] if i < len(is_swing_high) else False:
            last_swing_high.iloc[i] = dailyHigh22.iloc[i]
            lastSwingType.iloc[i] = "dailyHigh"
        if is_swing_low.iloc[i] if i < len(is_swing_low) else False:
            last_swing_low.iloc[i] = dailyLow22.iloc[i]
            lastSwingType.iloc[i] = "dailyLow"
    
    # Reindex to match df length
    bfvg_reindexed = bfvg.reindex(df.index, method='ffill').fillna(False)
    sfvg_reindexed = sfvg.reindex(df.index, method='ffill').fillna(False)
    lastSwingType_reindexed = lastSwingType.reindex(df.index, method='ffill').fillna("none")
    
    # Generate entry signals
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        bull_entry = bfvg_reindexed.iloc[i] and lastSwingType_reindexed.iloc[i] == "dailyLow"
        bear_entry = sfvg_reindexed.iloc[i] and lastSwingType_reindexed.iloc[i] == "dailyHigh"
        
        if bull_entry:
            ts = int(df.index[i].timestamp())
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if bear_entry:
            ts = int(df.index[i].timestamp())
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries