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

    if len(df) < 3:
        return entries

    # Extract OHLC data
    time = df['time'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    # Calculate ATR (Wilder ATR with length 144)
    tr = np.maximum(
        np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])),
        np.abs(low[1:] - close[:-1])
    )
    tr = np.concatenate([[tr[0]], tr])

    atr = np.zeros_like(tr)
    atr[0] = np.nan
    alpha = 1.0 / 144.0
    for i in range(1, len(atr)):
        if i < 144:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = atr[i-1] + alpha * (tr[i] - atr[i-1])

    atr = atr * 0.5  # fvgTH = 0.5

    # Calculate previous bar values using shift
    high_1 = np.roll(high, 1)
    low_1 = np.roll(low, 1)
    close_1 = np.roll(close, 1)
    high_2 = np.roll(high, 2)
    low_2 = np.roll(low, 2)

    high_1[0] = np.nan
    low_1[0] = np.nan
    close_1[0] = np.nan
    high_2[0] = np.nan
    high_2[1] = np.nan
    low_2[0] = np.nan
    low_2[1] = np.nan

    # Bullish/Bearish gap conditions
    bullG = (low > high_1) & ~np.isnan(high_1)
    bearG = (high < low_1) & ~np.isnan(low_1)

    bullG_1 = np.roll(bullG.astype(float), 1)
    bullG_1[0] = np.nan

    # Bullish FVG: (low - high[2]) > atr and low > high[2] and close[1] > high[2] and not (bullG or bullG[1])
    bull = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)):
        if np.isnan(atr[i]) or np.isnan(high_2[i]) or np.isnan(low[i]) or np.isnan(close_1[i]):
            continue
        if bullG[i] or (not np.isnan(bullG_1[i]) and bullG_1[i]):
            continue
        cond = (low[i] - high_2[i]) > atr[i] and low[i] > high_2[i] and close_1[i] > high_2[i]
        if cond:
            bull[i] = True

    # Bearish FVG: (low[2] - high) > atr and high < low[2] and close[1] < low[2] and not (bearG or bearG[1])
    bear = np.zeros(len(df), dtype=bool)
    bearG_1 = np.roll(bearG.astype(float), 1)
    bearG_1[0] = np.nan

    for i in range(2, len(df)):
        if np.isnan(atr[i]) or np.isnan(low_2[i]) or np.isnan(high[i]) or np.isnan(close_1[i]):
            continue
        if bearG[i] or (not np.isnan(bearG_1[i]) and bearG_1[i]):
            continue
        cond = (low_2[i] - high[i]) > atr[i] and high[i] < low_2[i] and close_1[i] < low_2[i]
        if cond:
            bear[i] = True

    # London session windows
    london_start_morning = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).replace(hour=6, minute=45, second=0, microsecond=0))
    london_end_morning = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).replace(hour=9, minute=45, second=0, microsecond=0))
    london_start_afternoon = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).replace(hour=14, minute=45, second=0, microsecond=0))
    london_end_afternoon = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).replace(hour=16, minute=45, second=0, microsecond=0))

    isWithinMorningWindow = (df['time'] >= london_start_morning.astype(np.int64) // 10**9) & (df['time'] < london_end_morning.astype(np.int64) // 10**9)
    isWithinAfternoonWindow = (df['time'] >= london_start_afternoon.astype(np.int64) // 10**9) & (df['time'] < london_end_afternoon.astype(np.int64) // 10**9)
    isWithinTimeWindow = isWithinMorningWindow | isWithinAfternoonWindow

    # Generate entries - FVG bullish = long, FVG bearish = short
    # Entries occur on confirmed bar (barstate.isconfirmed means close of current bar)
    for i in range(len(df)):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        direction = None
        if bull[i] and isWithinTimeWindow.iloc[i]:
            direction = 'long'
        elif bear[i] and isWithinTimeWindow.iloc[i]:
            direction = 'short'

        if direction:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries