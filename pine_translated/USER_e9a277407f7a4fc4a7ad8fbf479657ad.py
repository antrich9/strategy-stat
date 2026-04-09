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
    # Compute EMAs
    slowEma = df['close'].ewm(span=200, adjust=False).mean()
    fastEma = df['close'].ewm(span=50, adjust=False).mean()

    # Trend conditions
    upTrend = (df['close'] > slowEma) & (df['close'] > fastEma)
    downTrend = (df['close'] < slowEma) & (df['close'] < fastEma)

    # Candlestick patterns
    bullishEngulfing = (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
    bearishEngulfing = (df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & (df['open'] > df['close'].shift(1))

    # Pin bar detection
    upperWick = df['high'] - df[['open', 'close']].max(axis=1)
    lowerWick = df[['open', 'close']].min(axis=1) - df['low']
    bodyLength = (df['open'] - df['close']).abs()

    bullishPinBar = (lowerWick > bodyLength * 2) & (upperWick < bodyLength)
    bearishPinBar = (upperWick > bodyLength * 2) & (lowerWick < bodyLength)

    # Entry conditions
    longCondition = upTrend & (bullishEngulfing | bullishPinBar)
    shortCondition = downTrend & (bearishEngulfing | bearishPinBar)

    entries = []
    trade_num = 1

    # Iterate over bars
    for i in range(len(df)):
        # Skip bars where required indicators are NaN
        if pd.isna(slowEma.iloc[i]) or pd.isna(fastEma.iloc[i]):
            continue
        if pd.isna(bullishEngulfing.iloc[i]) or pd.isna(bearishEngulfing.iloc[i]) or pd.isna(bullishPinBar.iloc[i]) or pd.isna(bearishPinBar.iloc[i]):
            continue

        if longCondition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        if shortCondition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries