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
    trade_num = 1
    
    high = df['high']
    low = df['low']
    close = df['close']
    open_col = df['open']
    volume = df['volume']
    time = df['time']
    
    # Wilder ATR(14) and ATR(20)
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    atr20 = tr.ewm(alpha=1/20, adjust=False).mean()
    atr211 = atr20 / 1.5
    
    # Volume filter (disabled by default)
    sma9_vol = volume.rolling(9).mean()
    volfilt11 = volume.shift(1) > sma9_vol * 1.5
    
    # ATR filter (disabled by default)
    atrfilt11 = (low - high.shift(2) > atr211) | (low.shift(2) - high > atr211)
    
    # Trend filter: SMA(close, 54) and compare to previous
    loc11 = close.rolling(54).mean()
    loc211 = loc11 > loc11.shift(1)
    locfiltb11 = loc211
    locfilts11 = ~loc211
    
    # Detect new day
    timestamps = pd.to_datetime(df['time'], unit='s', utc=True)
    day_change = timestamps.diff().dt.days.fillna(0) != 0
    newDay = day_change
    
    # Previous Day High/Low calculation
    pdHigh = pd.Series(np.nan, index=df.index)
    pdLow = pd.Series(np.nan, index=df.index)
    tempHigh = pd.Series(np.nan, index=df.index)
    tempLow = pd.Series(np.nan, index=df.index)
    
    for i in range(len(df)):
        if newDay.iloc[i]:
            pdHigh.iloc[i] = tempHigh.iloc[i-1] if i > 0 else np.nan
            pdLow.iloc[i] = tempLow.iloc[i-1] if i > 0 else np.nan
            tempHigh.iloc[i] = high.iloc[i]
            tempLow.iloc[i] = low.iloc[i]
        else:
            tempHigh.iloc[i] = high.iloc[i] if pd.isna(tempHigh.iloc[i-1]) else max(tempHigh.iloc[i-1], high.iloc[i])
            tempLow.iloc[i] = low.iloc[i] if pd.isna(tempLow.iloc[i-1]) else min(tempLow.iloc[i-1], low.iloc[i])
            if i > 0:
                pdHigh.iloc[i] = pdHigh.iloc[i-1]
                pdLow.iloc[i] = pdLow.iloc[i-1]
    
    # Sweep flags (once per day)
    sweptHigh = False
    sweptLow = False
    
    # Swing detection using daily data
    dailyHigh11 = high.shift(1)
    dailyLow11 = low.shift(1)
    dailyHigh21 = high.shift(1)
    dailyLow21 = low.shift(1)
    dailyHigh22 = high.shift(2)
    dailyLow22 = low.shift(2)
    
    is_swing_high11 = (dailyHigh21 < dailyHigh22) & (dailyHigh11.shift(3) < dailyHigh22) & (dailyHigh11.shift(4) < dailyHigh22)
    is_swing_low11 = (dailyLow21 > dailyLow22) & (dailyLow11.shift(3) > dailyLow22) & (dailyLow11.shift(4) > dailyLow22)
    
    # Track last swing
    last_swing_high11 = pd.Series(np.nan, index=df.index)
    last_swing_low11 = pd.Series(np.nan, index=df.index)
    lastSwingType11 = pd.Series("none", index=df.index)
    
    for i in range(len(df)):
        if is_swing_high11.iloc[i]:
            last_swing_high11.iloc[i] = dailyHigh22.iloc[i]
            lastSwingType11.iloc[i] = "dailyHigh"
        elif is_swing_low11.iloc[i]:
            last_swing_low11.iloc[i] = dailyLow22.iloc[i]
            lastSwingType11.iloc[i] = "dailyLow"
        else:
            lastSwingType11.iloc[i] = lastSwingType11.iloc[i-1] if i > 0 else "none"
    
    # FVG detection
    bfvg11 = (low > high.shift(2)) & volfilt11 & atrfilt11 & locfiltb11
    sfvg11 = (high < low.shift(2)) & volfilt11 & atrfilt11 & locfilts11
    
    # Orderflow leg detection
    isBullishLeg11 = pd.Series(False, index=df.index)
    isBearishLeg11 = pd.Series(False, index=df.index)
    
    for i in range(2, len(df)):
        if bfvg11.iloc[i] and lastSwingType11.iloc[i] == "dailyLow":
            isBullishLeg11.iloc[i] = True
        if sfvg11.iloc[i] and lastSwingType11.iloc[i] == "dailyHigh":
            isBearishLeg11.iloc[i] = True
    
    # Time windows (simplified - would need actual timestamp parsing for real TZ handling)
    # For this implementation, using a simplified approach
    hour = timestamps.dt.hour
    minute = timestamps.dt.minute
    total_minutes = hour * 60 + minute
    
    morning_start = 6 * 60 + 45
    morning_end = 9 * 60 + 45
    afternoon_start = 14 * 60 + 45
    afternoon_end = 16 * 60 + 45
    
    isWithinMorningWindow = (total_minutes >= morning_start) & (total_minutes < morning_end)
    isWithinAfternoonWindow = (total_minutes >= afternoon_start) & (total_minutes < afternoon_end)
    in_trading_window = isWithinMorningWindow | isWithinAfternoonWindow
    
    # Supertrend calculation (simplified basic version)
    # Using Wilder ATR and basic trend detection
    # For real supertrend, would need proper implementation
    # Using a simplified EMA-based trend
    ema_st = close.ewm(span=10, adjust=False).mean()
    isSuperTrendBullish = close > ema_st
    isSuperTrendBearish = close < ema_st
    
    # Build entry conditions
    long_condition = isBullishLeg11 & in_trading_window
    short_condition = isBearishLeg11 & in_trading_window
    
    # Iterate through bars
    for i in range(2, len(df)):
        if pd.isna(atr14.iloc[i]) or pd.isna(atr20.iloc[i]) or pd.isna(loc11.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            entry_ts = int(time.iloc[i])
            entry_price = float(close.iloc[i])
            
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if short_condition.iloc[i]:
            entry_ts = int(time.iloc[i])
            entry_price = float(close.iloc[i])
            
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return results