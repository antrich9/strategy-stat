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
    trade_num = 0
    entries = []

    # Volume filter
    volfilt11 = df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5

    # ATR filter (Wilder)
    tr = np.maximum(df['high'] - df['low'], np.maximum(np.abs(df['high'] - df['close'].shift(1)), np.abs(df['low'] - df['close'].shift(1))))
    atr211 = tr.rolling(20).mean() / 1.5
    atrfilt11 = ((df['low'] - df['high'].shift(2) > atr211) | (df['low'].shift(2) - df['high'] > atr211))

    # Trend filter
    loc11 = df['close'].rolling(54).mean()
    loc211 = loc11 > loc11.shift(1)
    locfiltb11 = loc211
    locfilts11 = ~loc211

    # Bullish FVG: low > high[2]
    bfvg11 = (df['low'] > df['high'].shift(2)) & volfilt11 & atrfilt11 & locfiltb11

    # Bearish FVG: high < low[2]
    sfvg11 = (df['high'] < df['low'].shift(2)) & volfilt11 & atrfilt11 & locfilts11

    # Swing detection
    dailyHigh21 = df['high'].shift(1)
    dailyHigh22 = df['high'].shift(2)
    dailyHigh11 = df['high']
    dailyLow21 = df['low'].shift(1)
    dailyLow22 = df['low'].shift(2)
    dailyLow11 = df['low']

    is_swing_high11 = (dailyHigh21 < dailyHigh22) & (dailyHigh11.shift(3) < dailyHigh22) & (dailyHigh11.shift(4) < dailyHigh22)
    is_swing_low11 = (dailyLow21 > dailyLow22) & (dailyLow11.shift(3) > dailyLow22) & (dailyLow11.shift(4) > dailyLow22)

    lastSwingType11 = pd.Series("none", index=df.index)
    for i in range(1, len(df)):
        if is_swing_high11.iloc[i]:
            lastSwingType11.iloc[i] = "dailyHigh"
        else:
            lastSwingType11.iloc[i] = lastSwingType11.iloc[i-1]
        if is_swing_low11.iloc[i]:
            lastSwingType11.iloc[i] = "dailyLow"

    # Time window (UTC times converted from London times: London = UTC+0/1)
    df['ts'] = pd.to_datetime(df['time'], unit='s', utc=True)
    hours = df['ts'].dt.hour
    minutes = df['ts'].dt.minute
    total_minutes = hours * 60 + minutes
    morning_start = 6 * 60 + 45
    morning_end = 9 * 60 + 45
    afternoon_start = 14 * 60 + 45
    afternoon_end = 16 * 60 + 45
    isWithinMorningWindow = (total_minutes >= morning_start) & (total_minutes < morning_end)
    isWithinAfternoonWindow = (total_minutes >= afternoon_start) & (total_minutes < afternoon_end)
    in_trading_window = isWithinMorningWindow | isWithinAfternoonWindow

    # Entry conditions
    long_condition = bfvg11 & (lastSwingType11 == "dailyLow") & in_trading_window
    short_condition = sfvg11 & (lastSwingType11 == "dailyHigh") & in_trading_window

    # Generate entries
    for i in range(len(df)):
        if i < 5:
            continue
        if long_condition.iloc[i]:
            trade_num += 1
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
        elif short_condition.iloc[i]:
            trade_num += 1
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

    return entries