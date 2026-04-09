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
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('datetime')

    # Resample to higher timeframes
    weekly = df.resample('W').agg({'high': 'max', 'low': 'min'}).dropna()
    daily = df.resample('D').agg({'high': 'max', 'low': 'min'}).dropna()

    # Uptrend: higher highs and higher lows
    weeklyHigherHighs = weekly['high'] > weekly['high'].shift(1)
    weeklyHigherLows = weekly['low'] > weekly['low'].shift(1)
    weeklyUptrend = weeklyHigherHighs & weeklyHigherLows

    dailyHigherHighs = daily['high'] > daily['high'].shift(1)
    dailyHigherLows = daily['low'] > daily['low'].shift(1)
    dailyUptrend = dailyHigherHighs & dailyHigherLows

    # Downtrend: lower highs and lower lows
    weeklyLowerHighs = weekly['high'] < weekly['high'].shift(1)
    weeklyLowerLows = weekly['low'] < weekly['low'].shift(1)
    weeklyDowntrend = weeklyLowerHighs & weeklyLowerLows

    dailyLowerHighs = daily['high'] < daily['high'].shift(1)
    dailyLowerLows = daily['low'] < daily['low'].shift(1)
    dailyDowntrend = dailyLowerHighs & dailyLowerLows

    # 50-period SMA on close
    sma_50 = df['close'].rolling(50).mean()

    # Crossover and crossunder
    close_above_sma = df['close'] > sma_50
    prev_close_below_sma = df['close'].shift(1) <= sma_50.shift(1)
    crossover = close_above_sma & prev_close_below_sma

    close_below_sma = df['close'] < sma_50
    prev_close_above_sma = df['close'].shift(1) >= sma_50.shift(1)
    crossunder = close_below_sma & prev_close_above_sma

    # Align higher timeframe data to lower timeframe index
    weeklyUptrend_aligned = weeklyUptrend.reindex(df.index, method='ffill')
    dailyUptrend_aligned = dailyUptrend.reindex(df.index, method='ffill')
    weeklyDowntrend_aligned = weeklyDowntrend.reindex(df.index, method='ffill')
    dailyDowntrend_aligned = dailyDowntrend.reindex(df.index, method='ffill')

    # Entry conditions
    longCondition = weeklyUptrend_aligned & dailyUptrend_aligned & crossover
    shortCondition = weeklyDowntrend_aligned & dailyDowntrend_aligned & crossunder

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(sma_50.iloc[i]):
            continue
        entry_ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        entry_price_guess = float(df['close'].iloc[i])

        if longCondition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1

        if shortCondition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1

    return entries