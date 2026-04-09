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
    # Indicator parameters
    sensitivity = 150
    fastLength = 20
    slowLength = 40
    channelLength = 20
    mult = 2.0

    close = df['close']

    # MACD
    fastEMA = close.ewm(span=fastLength, adjust=False).mean()
    slowEMA = close.ewm(span=slowLength, adjust=False).mean()
    macd = fastEMA - slowEMA

    # t1 = (macd - macd[1]) * sensitivity
    t1 = (macd - macd.shift(1)) * sensitivity

    # trendUp = max(t1, 0)
    trendUp = t1.clip(lower=0)

    # Explosion line (Bollinger Band width)
    dev = mult * close.rolling(channelLength).std(ddof=0)
    e1 = 2 * dev

    # Track whether an entry has been made after the last explosion
    entryMade = pd.Series(False, index=df.index)

    entries = []
    trade_num = 0

    # Iterate over bars starting from the second bar
    for i in range(1, len(df)):
        # Reset entry flag when trend strength falls below the explosion line
        if trendUp.iloc[i] < e1.iloc[i]:
            entryMade.iloc[i] = False
        else:
            # Preserve the previous state
            entryMade.iloc[i] = entryMade.iloc[i - 1]

        # Entry logic: no active entry, trend crosses above explosion line
        if (not entryMade.iloc[i]) and (trendUp.iloc[i] > e1.iloc[i]) and (trendUp.iloc[i - 1] <= e1.iloc[i - 1]):
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()

            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            # Mark that an entry has been opened for this explosion
            entryMade.iloc[i] = True

    return entries