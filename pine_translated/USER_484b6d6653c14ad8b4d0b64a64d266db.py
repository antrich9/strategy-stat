import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Strategy parameters
    length = 100
    maxCupDepth = 0.30
    minHandleDepth = 0.08
    maxHandleDepth = 0.12
    minHandleDuration = 5
    handleLength = 20
    volumeMultiplier = 1.5
    proximityTo52WeekHigh = 0.80
    minHandleDurationInWeeks = 1

    # Date range: Jan 1, 2020 to Dec 31, 2023
    start_ts = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
    end_ts = datetime(2023, 12, 31, tzinfo=timezone.utc).timestamp()

    # Precompute indicators
    df['lowest_low'] = df['low'].rolling(length).min()
    df['fiftyTwoWeekHigh'] = df['high'].rolling(252).max()
    df['fiftyDayMA'] = df['close'].rolling(50).mean()
    df['averageVolume'] = df['volume'].rolling(50).mean()

    entries = []
    trade_num = 1

    # State variables
    inCup = False
    inHandle = False
    tradeEntered = False
    cupLow = np.nan
    cupHigh = np.nan
    cupStart = np.nan
    cupEnd = np.nan
    handleStart = np.nan
    entryPrice = np.nan
    buyPoint = np.nan

    for i in range(len(df)):
        row = df.iloc[i]
        ts = row['time']
        dateCondition = start_ts <= ts <= end_ts

        # Skip bars with NaN indicators
        if pd.isna(row['lowest_low']) or pd.isna(row['fiftyTwoWeekHigh']) or pd.isna(row['fiftyDayMA']) or pd.isna(row['averageVolume']):
            continue

        # Cup start condition
        if not inCup and not inHandle and not tradeEntered:
            if (row['low'] < row['lowest_low'] * (1 + maxCupDepth) and
                row['close'] > row['fiftyTwoWeekHigh'] * proximityTo52WeekHigh and
                dateCondition):
                cupLow = row['low']
                cupStart = i
                inCup = True
                tradeEntered = False

        # Cup continuation / end condition
        if inCup:
            if not (row['low'] > cupLow * (1 - maxCupDepth) and row['low'] <= cupLow * (1 + maxCupDepth)):
                inCup = False
                cupHigh = row['high']
                cupEnd = i
                inHandle = True
                handleStart = i
                buyPoint = cupHigh

        # Handle conditions
        if inHandle and row['low'] > row['fiftyDayMA'] and dateCondition:
            if i - cupEnd > handleLength:
                inHandle = False
            if row['low'] < cupHigh * (1 - maxHandleDepth):
                inHandle = False
            if i - cupEnd >= minHandleDuration:
                if not tradeEntered and (i - handleStart) >= minHandleDurationInWeeks * 5:
                    if (row['close'] > buyPoint and
                        row['volume'] > row['averageVolume'] * volumeMultiplier):
                        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                        entry_price = row['close']
                        entries.append({
                            'trade_num': trade_num,
                            'direction': 'long',
                            'entry_ts': int(ts),
                            'entry_time': entry_time,
                            'entry_price_guess': entry_price,
                            'exit_ts': 0,
                            'exit_time': '',
                            'exit_price_guess': 0.0,
                            'raw_price_a': entry_price,
                            'raw_price_b': entry_price
                        })
                        trade_num += 1
                        tradeEntered = True
                        entryPrice = row['close']
                        inHandle = False

    return entries