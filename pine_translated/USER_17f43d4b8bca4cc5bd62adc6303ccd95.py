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
    # Default input values from the Pine Script
    use_CoppockCurve = True
    signalLogicCoppockCurve = 'Zero line'  # 'Zero line' or 'Moving Average'
    lengthCoppockCurve = 10
    longRocLengthCoppockCurve = 14
    shortRocLengthCoppockCurve = 11
    lengthCoppockCurveMA = 10

    close = df['close']

    # Rate of Change (ROC)
    roc_long = (close / close.shift(longRocLengthCoppockCurve) - 1) * 100
    roc_short = (close / close.shift(shortRocLengthCoppockCurve) - 1) * 100

    # Weighted Moving Average (WMA)
    def wma(series: pd.Series, window: int) -> pd.Series:
        weights = np.arange(1, window + 1)
        def weighted_avg(x):
            return np.dot(x, weights) / weights.sum()
        return series.rolling(window).apply(weighted_avg, raw=True)

    # Coppock Curve
    coppock = wma(roc_long + roc_short, lengthCoppockCurve)

    # EMA of the Coppock Curve
    coppockMA = coppock.ewm(span=lengthCoppockCurveMA, adjust=False).mean()

    # Entry signals based on the script's logic
    if use_CoppockCurve:
        if signalLogicCoppockCurve == 'Zero line':
            entry_signal_long = coppock > 0
            entry_signal_short = coppock < 0
        else:  # Moving Average
            entry_signal_long = coppock > coppockMA
            entry_signal_short = coppock < coppockMA
    else:
        # When Coppell Curve is disabled, always generate entries
        entry_signal_long = pd.Series(True, index=df.index)
        entry_signal_short = pd.Series(True, index=df.index)

    # Build the list of entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        # Skip bars where the indicator is not yet available
        if pd.isna(coppock.iloc[i]):
            continue

        # Long entry
        if entry_signal_long.iloc[i]:
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

        # Short entry
        if entry_signal_short.iloc[i]:
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