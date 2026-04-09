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
    
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    timestamps = df['time']
    
    atrMultiplier = 3.0
    inp1_11 = False
    inp2_11 = False
    inp3_11 = False
    superTrendMultiplier = 3
    superTrendPeriod = 10
    
    # Wilder ATR
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    atr1 = atr
    
    # ATR for filter
    atr2 = (tr.ewm(alpha=1/20, adjust=False).mean()) / 1.5
    
    # Volume filter
    volfilt11 = volume.shift(1) > volume.rolling(9).mean() * 1.5
    
    # ATR filter
    atrfilt11 = ((low - high.shift(2) > atr2) | (low.shift(2) - high > atr2))
    
    # Trend filter
    loc11 = close.rolling(54).mean()
    loc211 = loc11 > loc11.shift(1)
    locfiltb11 = loc211
    locfilts11 = ~loc211
    
    # SuperTrend implementation
    hl2 = (high + low) / 2
    atr_st = tr.ewm(alpha=1/superTrendPeriod, adjust=False).mean()
    upperBand = hl2 + (superTrendMultiplier * atr_st)
    lowerBand = hl2 - (superTrendMultiplier * atr_st)
    
    superTrendDirection = pd.Series(0, index=close.index)
    prev_close = close.shift(1)
    prev_upper = upperBand.shift(1)
    prev_lower = lowerBand.shift(1)
    
    for i in range(1, len(close)):
        if pd.isna(superTrendDirection.iloc[i-1]):
            superTrendDirection.iloc[i] = 1
        elif superTrendDirection.iloc[i-1] == 1:
            if close.iloc[i] < prev_lower.iloc[i]:
                superTrendDirection.iloc[i] = -1
            else:
                superTrendDirection.iloc[i] = 1
        elif superTrendDirection.iloc[i-1] == -1:
            if close.iloc[i] > prev_upper.iloc[i]:
                superTrendDirection.iloc[i] = 1
            else:
                superTrendDirection.iloc[i] = -1
    
    isSuperTrendBullish = superTrendDirection == 1
    isSuperTrendBearish = superTrendDirection == -1
    
    # Swing detection
    dailyHigh11 = high
    dailyLow11 = low
    dailyHigh21 = high.shift(1)
    dailyLow21 = low.shift(1)
    dailyHigh22 = high.shift(2)
    dailyLow22 = low.shift(2)
    
    is_swing_high11 = (dailyHigh21 < dailyHigh22) & (dailyHigh11.shift(3) < dailyHigh22) & (dailyHigh11.shift(4) < dailyHigh22)
    is_swing_low11 = (dailyLow21 > dailyLow22) & (dailyLow11.shift(3) > dailyLow22) & (dailyLow11.shift(4) > dailyLow22)
    
    lastSwingType11 = pd.Series("none", index=close.index)
    for i in range(len(close)):
        if is_swing_high11.iloc[i]:
            lastSwingType11.iloc[i] = "dailyHigh"
        elif is_swing_low11.iloc[i]:
            lastSwingType11.iloc[i] = "dailyLow"
        else:
            if i > 0:
                lastSwingType11.iloc[i] = lastSwingType11.iloc[i-1]
    
    # Time window filtering
    london_offsets = []
    for ts in timestamps:
        dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
        month = dt_utc.month
        is_dst = (month >= 3 and month <= 10)
        offset = 1 if is_dst else 0
        london_hour = dt_utc.hour + offset
        london_minute = dt_utc.minute
        is_morning = (london_hour > 6 or (london_hour == 6 and london_minute >= 45)) and (london_hour < 9 or (london_hour == 9 and london_minute < 45))
        is_afternoon = (london_hour > 14 or (london_hour == 14 and london_minute >= 45)) and (london_hour < 16 or (london_hour == 16 and london_minute < 45))
        london_offsets.append(is_morning or is_afternoon)
    isWithinTimeWindow = pd.Series(london_offsets, index=timestamps.index)
    
    # FVG conditions
    bfvg11 = (low > high.shift(2)) & volfilt11 & atrfilt11 & locfiltb11
    sfvg11 = (high < low.shift(2)) & volfilt11 & atrfilt11 & locfilts11
    
    # Order flow tracking
    isBullishLeg11 = pd.Series(False, index=close.index)
    isBearishLeg11 = pd.Series(False, index=close.index)
    
    for i in range(len(close)):
        if bfvg11.iloc[i] and lastSwingType11.iloc[i] == "dailyLow":
            isBullishLeg11.iloc[i] = True
        elif sfvg11.iloc[i] and lastSwingType11.iloc[i] == "dailyHigh":
            isBearishLeg11.iloc[i] = True
        else:
            if i > 0:
                isBullishLeg11.iloc[i] = isBullishLeg11.iloc[i-1]
                isBearishLeg11.iloc[i] = isBearishLeg11.iloc[i-1]
    
    # Entry conditions
    longCond = bfvg11 & isBullishLeg11 & isSuperTrendBullish & isWithinTimeWindow
    shortCond = sfvg11 & isBearishLeg11 & isSuperTrendBearish & isWithinTimeWindow
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if longCond.iloc[i] and not pd.isna(close.iloc[i]):
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(timestamps.iloc[i]),
                'entry_time': datetime.fromtimestamp(timestamps.iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        elif shortCond.iloc[i] and not pd.isna(close.iloc[i]):
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(timestamps.iloc[i]),
                'entry_time': datetime.fromtimestamp(timestamps.iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
    
    return entries