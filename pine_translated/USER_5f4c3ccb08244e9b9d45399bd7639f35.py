import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Constants from Pine Script
    length = 100
    maxCupDepth = 0.30
    maxHandleDepth = 0.12
    minHandleDuration = 5
    handleLength = 20
    volumeMultiplier = 1.5
    proximityTo52WeekHigh = 0.80
    minHandleDurationInWeeks = 1

    # Backtest date range (inclusive)
    start_ts = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime(2023, 12, 31, tzinfo=timezone.utc).timestamp())

    df = df.copy()
    df['date_cond'] = (df['time'] >= start_ts) & (df['time'] <= end_ts)

    # Pre‑compute indicators
    df['lowest_low_100'] = df['low'].rolling(window=length).min()
    df['threshold_cup'] = df['lowest_low_100'] * (1 + maxCupDepth)
    df['52wh'] = df['high'].rolling(window=252).max()
    df['sma50_close'] = df['close'].rolling(window=50).mean()
    df['sma50_vol'] = df['volume'].rolling(window=50).mean()

    # State variables
    inCup = False
    inHandle = False
    tradeEntered = False
    cupLow = np.nan
    cupStart = -1
    cupEnd = -1
    cupHigh = np.nan
    handleStart = -1
    buyPoint = np.nan
    entryPrice = np.nan

    trade_num = 1
    entries = []

    for i in range(len(df)):
        low = df['low'].iloc[i]
        high = df['high'].iloc[i]
        close = df['close'].iloc[i]
        volume = df['volume'].iloc[i]
        date_cond = df['date_cond'].iloc[i]

        # ---- Cup formation start ----
        if not inCup and not inHandle:
            thr = df['threshold_cup'].iloc[i]
            wh = df['52wh'].iloc[i]
            if not np.isnan(thr) and not np.isnan(wh) and date_cond:
                if low < thr and close > wh * proximityTo52WeekHigh:
                    inCup = True
                    cupLow = low
                    cupStart = i
                    cupHigh = np.nan

        # ---- Cup to Handle transition (simplified) ----
        if inCup:
            cupHigh = high
            cupEnd = i
            inHandle = True
            handleStart = i
            buyPoint = cupHigh
            inCup = False

        # ---- Handle logic & entry condition ----
        if inHandle:
            # Handle too long → abort
            if (i - cupEnd) > handleLength:
                inHandle = False
                continue
            # Handle dips too deep → abort
            if low < cupHigh * (1 - maxHandleDepth):
                inHandle = False
                continue

            # Duration requirements
            if (i - cupEnd) >= minHandleDuration and (i - handleStart) >= minHandleDurationInWeeks * 5:
                if not tradeEntered and close > buyPoint and volume > df['sma50_vol'].iloc[i] * volumeMultiplier:
                    # Low must be above 50‑day MA
                    if low > df['sma50_close'].iloc[i]:
                        entryPrice = close
                        tradeEntered = True
                        inHandle = False

                        entry_ts = int(df['time'].iloc[i])
                        entries.append({
                            'trade_num': trade_num,
                            'direction': 'long',
                            'entry_ts': entry_ts,
                            'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                            'entry_price_guess': close,
                            'exit_ts': 0,
                            'exit_time': '',
                            'exit_price_guess': 0.0,
                            'raw_price_a': close,
                            'raw_price_b': close
                        })
                        trade_num += 1

    return entries