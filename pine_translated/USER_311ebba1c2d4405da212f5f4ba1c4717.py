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
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    
    # Resample to daily OHLCV
    daily = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Get daily data series with proper lookback
    dailyHigh = daily['high']
    dailyLow = daily['low']
    dailyOpen = daily['open']
    dailyClose = daily['close']
    volume = daily['volume']
    
    # Previous day data
    dailyHigh11 = dailyHigh.shift(0)  # current day high
    dailyLow11 = dailyLow.shift(0)    # current day low
    dailyHigh21 = dailyHigh.shift(1)  # 1 day ago high
    dailyLow21 = dailyLow.shift(1)    # 1 day ago low
    dailyHigh22 = dailyHigh.shift(2)  # 2 days ago high
    dailyLow22 = dailyLow.shift(2)    # 2 days ago low
    prevDayHigh11 = dailyHigh.shift(1)  # previous day high
    prevDayLow11 = dailyLow.shift(1)    # previous day low
    
    # Volume filter: volume[1] > sma(volume, 9) * 1.5
    vol_sma = volume.rolling(9).mean()
    volfilt = volume.shift(1) > vol_sma * 1.5
    
    # ATR Filter
    def wilder_atr(high, low, close, length=20):
        tr = pd.concat([high - low, 
                        (high - close.shift(1)).abs(), 
                        (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr
    
    atr211 = wilder_atr(dailyHigh, dailyLow, dailyClose, 20) / 1.5
    atrfilt = (dailyLow22 - dailyHigh22 > atr211) | (dailyLow22 - dailyHigh11 > atr211)
    
    # Trend Filter
    loc11 = dailyClose.rolling(54).mean()
    loc211 = loc11 > loc11.shift(1)
    locfiltb = loc211  # bullish trend filter
    locfilts = ~loc211  # bearish trend filter
    
    # FVG detection - Bullish: dailyLow11 > dailyHigh22
    bfvg = (dailyLow11 > dailyHigh22) & volfilt & atrfilt & locfiltb
    
    # FVG detection - Bearish: dailyHigh11 < dailyLow22
    sfvg = (dailyHigh11 < dailyLow22) & volfilt & atrfilt & locfilts
    
    # Swing detection
    is_swing_high = (dailyHigh21 < dailyHigh22) & (dailyHigh.shift(3) < dailyHigh22) & (dailyHigh.shift(4) < dailyHigh22)
    is_swing_low = (dailyLow21 > dailyLow22) & (dailyLow.shift(3) > dailyLow22) & (dailyLow.shift(4) > dailyLow22)
    
    # Track last swing type
    lastSwingType = pd.Series(index=daily.index, dtype='object')
    last_swing_high = pd.Series(np.nan, index=daily.index)
    last_swing_low = pd.Series(np.nan, index=daily.index)
    
    current_swing_type = "none"
    for i in range(len(daily)):
        if is_swing_high.iloc[i]:
            last_swing_high.iloc[i] = dailyHigh22.iloc[i]
            last_swing_low.iloc[i] = last_swing_low.iloc[i-1] if i > 0 else np.nan
            current_swing_type = "dailyHigh"
        elif is_swing_low.iloc[i]:
            last_swing_low.iloc[i] = dailyLow22.iloc[i]
            last_swing_high.iloc[i] = last_swing_high.iloc[i-1] if i > 0 else np.nan
            current_swing_type = "dailyLow"
        else:
            last_swing_high.iloc[i] = last_swing_high.iloc[i-1] if i > 0 else np.nan
            last_swing_low.iloc[i] = last_swing_low.iloc[i-1] if i > 0 else np.nan
        lastSwingType.iloc[i] = current_swing_type
    
    # Determine daily entry signals
    bullishFVG = bfvg & (lastSwingType == "dailyLow")
    bearishFVG = sfvg & (lastSwingType == "dailyHigh")
    
    # Map daily signals back to original timeframe
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        ts = df.index[i]
        daily_ts = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Find corresponding daily index
        daily_idx = daily.index.get_indexer([daily_ts], method='ffill')[0]
        if daily_idx < 0:
            daily_idx = 0
        
        entry_price = df['close'].iloc[i]
        
        # Check bullish entry
        if daily_idx < len(daily) and bullishFVG.iloc[daily_idx]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts.timestamp()),
                'entry_time': ts.isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        # Check bearish entry
        if daily_idx < len(daily) and bearishFVG.iloc[daily_idx]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts.timestamp()),
                'entry_time': ts.isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries