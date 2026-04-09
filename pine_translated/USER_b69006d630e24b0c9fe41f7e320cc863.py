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
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    rangeKUS = 30
    priceSmoothingKUS = 0.3
    indexSmoothingKUS = 0.3
    emaLengthKUS = 20
    useKUS = True
    signalTypeKUS = '0 Line + MA Signal'
    inverseKUS = False
    crossOnlyKUS = True
    
    highestHighKUS = high.rolling(window=rangeKUS, min_periods=rangeKUS).max()
    lowestLowKUS = low.rolling(window=rangeKUS, min_periods=rangeKUS).min()
    greatestRangeKUS = highestHighKUS - lowestLowKUS
    
    midPriceKUS = (high + low) / 2.0
    
    priceLocationKUS_raw = 2 * ((midPriceKUS - lowestLowKUS) / np.maximum(greatestRangeKUS, 0.001)) - 1
    
    smoothedLocationKUS = np.zeros(len(df))
    smoothedLocationKUS[0] = priceLocationKUS_raw.iloc[0] if not np.isnan(priceLocationKUS_raw.iloc[0]) else 0.0
    for i in range(1, len(df)):
        pl = priceLocationKUS_raw.iloc[i]
        if np.isnan(pl):
            pl = 0.0
        smoothedLocationKUS[i] = priceSmoothingKUS * smoothedLocationKUS[i-1] + (1 - priceSmoothingKUS) * pl
    
    smoothedLocationKUS = np.clip(smoothedLocationKUS, -0.99, 0.99)
    smoothedLocationKUS_series = pd.Series(smoothedLocationKUS)
    
    fishIndexKUS_arr = np.zeros(len(df))
    for i in range(len(df)):
        sl = smoothedLocationKUS[i]
        if sl >= 1.0:
            sl = 0.999
        elif sl <= -1.0:
            sl = -0.999
        fishIndexKUS_arr[i] = np.log((1 + sl) / (1 - sl))
    fishIndexKUS = pd.Series(fishIndexKUS_arr)
    
    smoothedFishKUS = np.zeros(len(df))
    smoothedFishKUS[0] = fishIndexKUS_arr[0] if not np.isnan(fishIndexKUS_arr[0]) else 0.0
    for i in range(1, len(df)):
        fi = fishIndexKUS_arr[i]
        if np.isnan(fi):
            fi = 0.0
        smoothedFishKUS[i] = indexSmoothingKUS * smoothedFishKUS[i-1] + (1 - indexSmoothingKUS) * fi
    
    smoothedFishKUS_series = pd.Series(smoothedFishKUS)
    maKUS = smoothedFishKUS_series.ewm(span=emaLengthKUS, adjust=False).mean()
    
    if signalTypeKUS == '0 Line + MA Signal':
        signalEntryLongKUS = (smoothedFishKUS_series > 0) & (smoothedFishKUS_series > maKUS)
        signalEntryShortKUS = (smoothedFishKUS_series < 0) & (smoothedFishKUS_series < maKUS)
    elif signalTypeKUS == '0 Line':
        signalEntryLongKUS = smoothedFishKUS_series > 0
        signalEntryShortKUS = smoothedFishKUS_series < 0
    else:
        signalEntryLongKUS = smoothedFishKUS_series > maKUS
        signalEntryShortKUS = smoothedFishKUS_series < maKUS
    
    if crossOnlyKUS:
        prev_long = signalEntryLongKUS.shift(1).fillna(False)
        prev_short = signalEntryShortKUS.shift(1).fillna(False)
        signalEntryLongKUS = signalEntryLongKUS & ~prev_long
        signalEntryShortKUS = signalEntryShortKUS & ~prev_short
    
    if inverseKUS:
        finalLongSignalKUS = signalEntryShortKUS
        finalShortSignalKUS = signalEntryLongKUS
    else:
        finalLongSignalKUS = signalEntryLongKUS
        finalShortSignalKUS = signalEntryShortKUS
    
    trade_num = 1
    entries = []
    
    for i in range(len(df)):
        if finalLongSignalKUS.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        elif finalShortSignalKUS.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    return entries