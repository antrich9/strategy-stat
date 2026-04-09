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
    close = df['close']
    high = df['high']
    low = df['low']

    # Calculate EMAs
    fastEMA = close.ewm(span=8, adjust=False).mean()
    mediumEMA = close.ewm(span=20, adjust=False).mean()
    slowEMA = close.ewm(span=50, adjust=False).mean()

    # Calculate ATR (Wilder's method)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()

    # Time filter
    times = pd.to_datetime(df['time'], unit='s', utc=True)
    hour = times.dt.hour
    in_trading_window = ((hour >= 7) & (hour < 10)) | ((hour >= 14) & (hour < 17))

    # Trend confirmation
    isBullishTrend = (close > slowEMA) & (fastEMA > mediumEMA) & (mediumEMA > slowEMA)
    isBearishTrend = (close < slowEMA) & (fastEMA < mediumEMA) & (mediumEMA < slowEMA)

    # Entry conditions using crossover/crossunder
    longCondition = (close > fastEMA) & (close.shift(1) <= fastEMA.shift(1)) & isBullishTrend & in_trading_window
    shortCondition = (close < fastEMA) & (close.shift(1) >= fastEMA.shift(1)) & isBearishTrend & in_trading_window

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(fastEMA.iloc[i]) or pd.isna(mediumEMA.iloc[i]) or pd.isna(slowEMA.iloc[i]) or pd.isna(atr.iloc[i]):
            continue
        if longCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        elif shortCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1

    return entries