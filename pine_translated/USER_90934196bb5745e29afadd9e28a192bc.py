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
    trade_num = 0

    # HTF = 240min (4H), LTF = 15min
    htf_period = 240
    ltf_period = 15

    # Compute HTF (240min) OHLC aligned to each 1min bar
    htf_high_2 = df['high'].rolling(htf_period * 2).max().shift(1)
    htf_low_1 = df['low'].rolling(htf_period, min_periods=htf_period).min().shift(1)
    htf_high_1 = df['high'].rolling(htf_period, min_periods=htf_period).max().shift(1)
    htf_low_2 = df['low'].rolling(htf_period * 2).min().shift(2)

    # Compute LTF (15min) OHLC aligned to each 1min bar
    ltf_high_2 = df['high'].rolling(ltf_period * 2).max().shift(1)
    ltf_low_1 = df['low'].rolling(ltf_period, min_periods=ltf_period).min().shift(1)
    ltf_high = df['high'].rolling(ltf_period, min_periods=ltf_period).max()
    ltf_low = df['low'].rolling(ltf_period, min_periods=ltf_period).min()
    ltf_low_2 = df['low'].rolling(ltf_period * 2).min().shift(2)

    # HTF Bullish FVG: htf_low[1] > htf_high[2]
    htf_bull_fvg = htf_low_1 > htf_high_2

    # HTF Bearish FVG: htf_high[1] < htf_low[2]
    htf_bear_fvg = htf_high_1 < htf_low_2

    # LTF Bullish FVG: ltf_low > ltf_high_2
    ltf_bull_fvg = ltf_low > ltf_high_2

    # LTF Bearish FVG: ltf_high < ltf_low_2
    ltf_bear_fvg = ltf_high < ltf_low_2

    # London session windows
    dt_series = pd.to_datetime(df['time'], unit='s', utc=True)
    hour = dt_series.dt.hour
    minute = dt_series.dt.minute
    total_minutes = hour * 60 + minute

    is_in_window1 = (total_minutes >= 465) & (total_minutes < 585)  # 07:45 - 09:45
    is_in_window2 = (total_minutes >= 840) & (total_minutes < 1005)  # 14:00 - 16:45
    in_trading_window = is_in_window1 | is_in_window2

    # Long entry: HTF bull FVG + LTF bull FVG + London session
    long_condition = htf_bull_fvg & ltf_bull_fvg & in_trading_window

    # Short entry: HTF bear FVG + LTF bear FVG + London session
    short_condition = htf_bear_fvg & ltf_bear_fvg & in_trading_window

    n = len(df)

    for i in range(n):
        # Skip bars with NaN in required indicators
        if pd.isna(htf_bull_fvg.iloc[i]) or pd.isna(ltf_bull_fvg.iloc[i]) or pd.isna(htf_bear_fvg.iloc[i]) or pd.isna(ltf_bear_fvg.iloc[i]):
            continue
        if pd.isna(htf_low_1.iloc[i]) or pd.isna(htf_high_2.iloc[i]):
            continue

        if long_condition.iloc[i]:
            trade_num += 1
            ts = int(df['time'].iloc[i])
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

        if short_condition.iloc[i]:
            trade_num += 1
            ts = int(df['time'].iloc[i])
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

    return entries