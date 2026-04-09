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
    
    useT3 = True
    factorT3 = 0.7
    signalTypeT3 = 'MA + Price'
    crossOnlyT3 = True
    
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    def calc_t3(src, length, factor):
        e1 = src.ewm(span=length, adjust=False).mean()
        e2 = e1.ewm(span=length, adjust=False).mean()
        e3 = e2.ewm(span=length, adjust=False).mean()
        e4 = e3.ewm(span=length, adjust=False).mean()
        e5 = e4.ewm(span=length, adjust=False).mean()
        e6 = e5.ewm(span=length, adjust=False).mean()
        c1 = -factor ** 3
        c2 = 3 * factor ** 2 + 3 * factor ** 3
        c3 = -6 * factor ** 2 - 3 * factor - 3 * factor ** 3
        c4 = 1 + 3 * factor + factor ** 2 + 3 * factor ** 2
        return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    
    t3_25 = calc_t3(df['close'], 25, factorT3)
    t3_100 = calc_t3(df['close'], 100, factorT3)
    t3_200 = calc_t3(df['close'], 200, factorT3)
    
    longConditionIndiT3 = (df['close'] > t3_25) & (df['close'] > t3_100) & (df['close'] > t3_200)
    shortConditionIndiT3 = (df['close'] < t3_25) & (df['close'] < t3_100) & (df['close'] < t3_200)
    longConditionIndiT3MA = (t3_100 < t3_25) & (t3_200 < t3_100)
    shortConditionIndiT3MA = (t3_100 > t3_25) & (t3_200 > t3_100)
    
    if signalTypeT3 == 'MA + Price':
        signalEntryLongT3 = longConditionIndiT3 & longConditionIndiT3MA
        signalEntryShortT3 = shortConditionIndiT3 & shortConditionIndiT3MA
    elif signalTypeT3 == 'MA Only':
        signalEntryLongT3 = longConditionIndiT3MA
        signalEntryShortT3 = shortConditionIndiT3MA
    else:
        signalEntryLongT3 = longConditionIndiT3
        signalEntryShortT3 = shortConditionIndiT3
    
    if crossOnlyT3:
        finalLongSignalT3 = signalEntryLongT3 & ~signalEntryLongT3.shift(1).fillna(False)
        finalShortSignalT3 = signalEntryShortT3 & ~signalEntryShortT3.shift(1).fillna(False)
    else:
        finalLongSignalT3 = signalEntryLongT3
        finalShortSignalT3 = signalEntryShortT3
    
    pct_change = df['close'].diff(trendilo_smooth) / df['close'] * 100
    m = trendilo_sigma * 0.6
    offset_val = int(trendilo_offset * (trendilo_length - 1))
    def alma_weights(n, offset, sigma):
        wts = np.zeros(n)
        for i in range(n):
            wts[i] = np.exp(-((i - offset) ** 2) / (2 * sigma ** 2))
        return wts / wts.sum()
    alma_wts = alma_weights(trendilo_length, offset_val, m)
    avg_pct_change = pct_change.rolling(trendilo_length).apply(lambda x: np.sum(alma_wts * x.values[::-1]), raw=True)
    rms = trendilo_bmult * np.sqrt((avg_pct_change ** 2).rolling(trendilo_length).mean())
    
    trend_dir = pd.Series(0, index=df.index)
    trend_dir[avg_pct_change > rms] = 1
    trend_dir[avg_pct_change < -rms] = -1
    
    long_condition = finalLongSignalT3 & (trend_dir == 1)
    short_condition = finalShortSignalT3 & (trend_dir == -1)
    
    finalLongEntry = long_condition & ~long_condition.shift(1).fillna(False)
    finalShortEntry = short_condition & ~short_condition.shift(1).fillna(False)
    
    for i in range(len(df)):
        if finalLongEntry.iloc[i]:
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif finalShortEntry.iloc[i]:
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries