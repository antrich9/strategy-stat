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

    # Time window parameters
    london_start_morning_hour = 6
    london_start_morning_minute = 45
    london_end_morning_hour = 9
    london_end_morning_minute = 45

    london_start_afternoon_hour = 14
    london_start_afternoon_minute = 45
    london_end_afternoon_hour = 16
    london_end_afternoon_minute = 45

    # Fair Value Gap Width Filter
    fvg_th = 0.5

    # ATR calculation (Wilder)
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/144, adjust=False).mean() * fvg_th

    # Bullish/Bearish gap detection
    bullG = low > high.shift(1)
    bearG = high < low.shift(1)

    # Bullish FVG condition
    bull_cond = (low - high.shift(2)) > atr
    bull_cond = bull_cond & (low > high.shift(2))
    bull_cond = bull_cond & (close.shift(1) > high.shift(2))
    bull_cond = bull_cond & ~(bullG | bullG.shift(1))

    # Bearish FVG condition
    bear_cond = (low.shift(2) - high) > atr
    bear_cond = bear_cond & (high < low.shift(2))
    bear_cond = bear_cond & (close.shift(1) < low.shift(2))
    bear_cond = bear_cond & ~(bearG | bearG.shift(1))

    # Time window check
    isWithinMorningWindow = pd.Series(False, index=df.index)
    isWithinAfternoonWindow = pd.Series(False, index=df.index)

    for i in df.index:
        ts = df.loc[i, 'time']
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        total_minutes = hour * 60 + minute

        morning_start = london_start_morning_hour * 60 + london_start_morning_minute
        morning_end = london_end_morning_hour * 60 + london_end_morning_minute
        afternoon_start = london_start_afternoon_hour * 60 + london_start_afternoon_minute
        afternoon_end = london_end_afternoon_hour * 60 + london_end_afternoon_minute

        if morning_start <= total_minutes < morning_end:
            isWithinMorningWindow.loc[i] = True
        if afternoon_start <= total_minutes < afternoon_end:
            isWithinAfternoonWindow.loc[i] = True

    in_trading_window = isWithinMorningWindow | isWithinAfternoonWindow

    # Final entry conditions
    long_entries = bull_cond & in_trading_window
    short_entries = bear_cond & in_trading_window

    # Build entries list
    for i in df.index:
        if long_entries.loc[i]:
            ts = int(df.loc[i, 'time'])
            entry_price = df.loc[i, 'close']
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_entries.loc[i]:
            ts = int(df.loc[i, 'time'])
            entry_price = df.loc[i, 'close']
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries