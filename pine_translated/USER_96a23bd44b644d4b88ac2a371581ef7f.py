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
    results = []
    trade_num = 1

    # Extract hour from timestamp
    df = df.copy()
    df['hour'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.hour

    # Asian Session settings
    asianStartHour = 8
    asianEndHour = 12

    # Detect Asian Session
    inAsianSession = (df['hour'] >= asianStartHour) & (df['hour'] < asianEndHour)
    asianSessionEnded = ~inAsianSession & inAsianSession.shift(1).fillna(False)

    # Asian session high/low tracking
    asianHigh = np.nan
    asianLow = np.nan
    tempHigh = np.nan
    tempLow = np.nan

    # Sweep tracking
    sweptHigh = False
    sweptLow = False

    # Order block state tracking
    bullOBActive = False
    bearOBActive = False
    bullBoxTop = np.nan
    bullBoxBottom = np.nan
    bearBoxTop = np.nan
    bearBoxBottom = np.nan

    # Tap counters
    bulltap1 = 0
    beartap1 = 0

    # Previous values for engulfing detection
    prevOpen = df['open'].shift(1)
    prevClose = df['close'].shift(1)

    # Bullish engulfing: prev bar bearish (prevOpen > prevClose) and current close > prevOpen
    isBullEngulf = (prevOpen > prevClose) & (df['close'] > prevOpen)

    # Bearish engulfing: prev bar bullish (prevOpen < prevClose) and current close < prevOpen
    isBearEngulf = (prevOpen < prevClose) & (df['close'] < prevOpen)

    # Iterate through bars
    for i in range(1, len(df)):
        row = df.iloc[i]

        # Update Asian session tracking
        if inAsianSession.iloc[i]:
            tempHigh = row['high'] if np.isnan(tempHigh) else max(tempHigh, row['high'])
            tempLow = row['low'] if np.isnan(tempLow) else min(tempLow, row['low'])

        # End of Asian session - set high/low and reset sweep flags
        if asianSessionEnded.iloc[i]:
            if not np.isnan(tempHigh):
                asianHigh = tempHigh
            if not np.isnan(tempLow):
                asianLow = tempLow
            tempHigh = np.nan
            tempLow = np.nan
            sweptHigh = False
            sweptLow = False
            bullOBActive = False
            bearOBActive = False
            bullBoxTop = np.nan
            bullBoxBottom = np.nan
            bearBoxTop = np.nan
            bearBoxBottom = np.nan

        # Sweep detection
        if not sweptHigh and not np.isnan(asianHigh) and row['high'] > asianHigh:
            sweptHigh = True

        if not sweptLow and not np.isnan(asianLow) and row['low'] < asianLow:
            sweptLow = True

        # Bullish engulfing detected
        if isBullEngulf.iloc[i]:
            bullOBActive = True
            bulltap1 = 0
            bullBoxTop = prevOpen.iloc[i]
            bullBoxBottom = min(prevClose.iloc[i], row['low'])
            if prevClose.iloc[i] < prevOpen.iloc[i]:
                if row['low'] < prevClose.iloc[i]:
                    bullBoxBottom = row['low']
                else:
                    bullBoxBottom = prevClose.iloc[i]

        # Bearish engulfing detected
        if isBearEngulf.iloc[i]:
            bearOBActive = True
            beartap1 = 0
            bearBoxTop = max(prevOpen.iloc[i], row['high'])
            bearBoxBottom = prevOpen.iloc[i]
            if prevClose.iloc[i] > prevOpen.iloc[i]:
                if row['high'] > prevOpen.iloc[i]:
                    bearBoxTop = row['high']
                else:
                    bearBoxTop = prevOpen.iloc[i]

        # Long entry: price taps bull OB bottom (price crosses below bullBoxBottom)
        if bullOBActive and not np.isnan(bullBoxBottom):
            if i > 0 and df['close'].iloc[i-1] >= bullBoxBottom and row['low'] < bullBoxBottom:
                entry_ts = int(row['time'])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                results.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': row['close'],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': row['close'],
                    'raw_price_b': row['close']
                })
                trade_num += 1
                bullOBActive = False

        # Short entry: price taps bear OB top (price crosses above bearBoxTop)
        if bearOBActive and not np.isnan(bearBoxTop):
            if i > 0 and df['close'].iloc[i-1] <= bearBoxTop and row['high'] > bearBoxTop:
                entry_ts = int(row['time'])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                results.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': row['close'],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': row['close'],
                    'raw_price_b': row['close']
                })
                trade_num += 1
                bearOBActive = False

    return results