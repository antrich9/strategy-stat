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
    time = df['time']

    # T3 parameters
    factorT3 = 0.7
    srcT3 = close

    # T3 calculation function
    def gdT3(src, length):
        e1 = src.ewm(span=length, adjust=False).mean()
        e2 = e1.ewm(span=length, adjust=False).mean()
        e3 = e2.ewm(span=length, adjust=False).mean()
        e4 = e3.ewm(span=length, adjust=False).mean()
        e5 = e4.ewm(span=length, adjust=False).mean()
        e6 = e5.ewm(span=length, adjust=False).mean()
        c1 = -factorT3 * factorT3 * factorT3
        c2 = 3 * factorT3 * factorT3 + 3 * factorT3 * factorT3 * factorT3
        c3 = -6 * factorT3 * factorT3 - 3 * factorT3 - 3 * factorT3 * factorT3 * factorT3
        c4 = 1 + 3 * factorT3 + factorT3 * factorT3 * factorT3 + 3 * factorT3 * factorT3
        return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    t3_25 = gdT3(srcT3, 25)
    t3_100 = gdT3(srcT3, 100)
    t3_200 = gdT3(srcT3, 200)

    # Price conditions
    longConditionIndiT3 = (close > t3_25) & (close > t3_100) & (close > t3_200)
    shortConditionIndiT3 = (close < t3_25) & (close < t3_100) & (close < t3_200)

    # MA conditions
    longConditionIndiT3MA = (t3_100 < t3_25) & (t3_200 < t3_100)
    shortConditionIndiT3MA = (t3_100 > t3_25) & (t3_200 > t3_100)

    # Use MA + Price for both based on defaults
    signalEntryLongT3 = longConditionIndiT3 & longConditionIndiT3MA
    signalEntryShortT3 = shortConditionIndiT3 & shortConditionIndiT3MA

    # Cross only: only trigger on first appearance (transition from false to true)
    signalEntryLongT3_cross = signalEntryLongT3 & ~signalEntryLongT3.shift(1).fillna(False)
    signalEntryShortT3_cross = signalEntryShortT3 & ~signalEntryShortT3.shift(1).fillna(False)

    # No inverse
    finalLongSignalT3 = signalEntryLongT3_cross
    finalShortSignalT3 = signalEntryShortT3_cross

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(t3_25.iloc[i]) or pd.isna(t3_100.iloc[i]) or pd.isna(t3_200.iloc[i]):
            continue

        ts = int(time.iloc[i])
        price = close.iloc[i]

        if finalLongSignalT3.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1

        if finalShortSignalT3.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1

    return entries