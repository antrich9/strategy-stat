import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Ensure data is sorted by time
    df = df.sort_values('time').reset_index(drop=True)

    # Strategy parameters
    fastLength = 8
    mediumLength = 20
    slowLength = 50
    rsi_length = 14
    atr_length = 14

    # Date range (Unix timestamps in seconds)
    start_ts = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime(2023, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp())

    # Pip size (default for non‑JPY pairs)
    pipSize = 0.0002

    # Price series
    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']

    # EMAs (Exponential Moving Average – Wilder style)
    fastEMA = close.ewm(span=fastLength, adjust=False).mean()
    mediumEMA = close.ewm(span=mediumLength, adjust=False).mean()
    slowEMA = close.ewm(span=slowLength, adjust=False).mean()

    # Wilder RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / rsi_length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_length, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Wilder ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_length, adjust=False).mean()

    # Price‑action thresholds
    upperThreshold = high - (high - low) * 0.31
    lowerThreshold = low + (high - low) * 0.31

    # Candle patterns
    bullishCandle = (close > upperThreshold) & (open_ > upperThreshold) & (low <= fastEMA)
    bearishCandle = (close < lowerThreshold) & (open_ < lowerThreshold) & (high >= fastEMA)

    # EMA alignment
    longEMAsAligned = (fastEMA > mediumEMA) & (mediumEMA > slowEMA)
    shortEMAsAligned = (fastEMA < mediumEMA) & (mediumEMA < slowEMA)

    # RSI conditions
    rsi_lt_70 = rsi < 70
    rsi_gt_30 = rsi > 30

    # Combined entry conditions
    longCondition = bullishCandle & longEMAsAligned & rsi_lt_70
    shortCondition = bearishCandle & shortEMAsAligned & rsi_gt_30

    # Date‑range filter
    inDateRange = (df['time'] >= start_ts) & (df['time'] <= end_ts)
    longCondition = longCondition & inDateRange
    shortCondition = shortCondition & inDateRange

    # Generate entry records
    entries = []
    trade_num = 1

    for i in range(len(df)):
        # Skip bars with any NaN indicator value
        if (pd.isna(fastEMA.iloc[i]) or pd.isna(mediumEMA.iloc[i]) or
            pd.isna(slowEMA.iloc[i]) or pd.isna(rsi.iloc[i]) or pd.isna(atr.iloc[i])):
            continue

        if longCondition.iloc[i]:
            entry_price = high.iloc[i] + pipSize
            entry_ts = int(df['time'].iloc[i])
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
            trade_num += 1

        elif shortCondition.iloc[i]:
            entry_price = low.iloc[i] - pipSize
            entry_ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries