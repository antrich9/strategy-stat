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
    # Input parameters (from Pine Script)
    useHullMA = True
    usecolorHullMA = True
    lengthHullMA = 9
    srcHullMA_col = 'close'

    useT3 = True
    crossT3 = True
    inverseT3 = False
    lengthT3 = 5
    factorT3 = 0.7
    highlightMovementsT3 = True
    srcT3_col = 'close'

    pinBarSize = 30

    # Calculate Hull MA
    half_length = int(lengthHullMA / 2)
    wma1 = df[srcHullMA_col].ewm(span=half_length, adjust=False).mean()
    wma2 = df[srcHullMA_col].ewm(span=lengthHullMA, adjust=False).mean()
    hullmaHullMA = (2 * wma1 - wma2).ewm(span=int(np.sqrt(lengthHullMA)), adjust=False).mean()
    hullmaHullMA_prev = hullmaHullMA.shift(1)
    sigHullMA = (hullmaHullMA > hullmaHullMA_prev).astype(int) - (hullmaHullMA < hullmaHullMA_prev).astype(int)

    # Calculate T3
    def calc_t3(series, length, factor):
        ema1 = series.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factor) - ema2 * factor

    t3 = calc_t3(calc_t3(calc_t3(df[srcT3_col], lengthT3, factorT3), lengthT3, factorT3), lengthT3, factorT3)
    t3_prev = t3.shift(1)
    t3Signals = (t3 > t3_prev).astype(int) - (t3 < t3_prev).astype(int)

    # Hull MA long/short signals
    if useHullMA:
        if usecolorHullMA:
            signalHullMALong = (sigHullMA > 0) & (df['close'] > hullmaHullMA)
            signalHullMAShort = (sigHullMA < 0) & (df['close'] < hullmaHullMA)
        else:
            signalHullMALong = df['close'] > hullmaHullMA
            signalHullMAShort = df['close'] < hullmaHullMA
    else:
        signalHullMALong = pd.Series(True, index=df.index)
        signalHullMAShort = pd.Series(True, index=df.index)

    # T3 long signals
    basicLongCondition = (t3Signals > 0) & (df['close'] > t3)
    if useT3:
        if highlightMovementsT3:
            t3SignalsLong = basicLongCondition
        else:
            t3SignalsLong = df['close'] > t3
    else:
        t3SignalsLong = pd.Series(True, index=df.index)

    t3SignalsLong_prev = t3SignalsLong.shift(1)
    if crossT3:
        t3SignalsLongCross = (~t3SignalsLong_prev) & t3SignalsLong
    else:
        t3SignalsLongCross = t3SignalsLong

    if inverseT3:
        t3SignalsLongFinal = ~t3SignalsLongCross
    else:
        t3SignalsLongFinal = t3SignalsLongCross

    # T3 short signals
    basicShortCondition = (t3Signals < 0) & (df['close'] < t3)
    if useT3:
        if highlightMovementsT3:
            t3SignalsShort = basicShortCondition
        else:
            t3SignalsShort = df['close'] < t3
    else:
        t3SignalsShort = pd.Series(True, index=df.index)

    t3SignalsShort_prev = t3SignalsShort.shift(1)
    if crossT3:
        t3SignalsShortCross = (~t3SignalsShort_prev) & t3SignalsShort
    else:
        t3SignalsShortCross = t3SignalsShort

    if inverseT3:
        t3SignalsShortFinal = ~t3SignalsShortCross
    else:
        t3SignalsShortFinal = t3SignalsShortCross

    # Pin bar calculations
    bodySize = (df['close'] - df['open']).abs()
    open_gt_close = df['open'] > df['close']
    lowerWickSize = np.where(open_gt_close, df['close'] - df['low'], df['open'] - df['low'])
    upperWickSize = np.where(open_gt_close, df['high'] - df['open'], df['high'] - df['close'])

    lowerWickSize = pd.Series(lowerWickSize, index=df.index)
    upperWickSize = pd.Series(upperWickSize, index=df.index)

    isBullishPinBar = (df['close'] > df['open']) & \
                      (lowerWickSize > bodySize * pinBarSize / 100) & \
                      (lowerWickSize > upperWickSize * 2)

    isBearishPinBar = (df['close'] < df['open']) & \
                      (upperWickSize > bodySize * pinBarSize / 100) & \
                      (upperWickSize > lowerWickSize * 2)

    # Combined entry conditions
    entryCondition = signalHullMALong & t3SignalsLongFinal & isBullishPinBar
    entryConditionShort = signalHullMAShort & t3SignalsShortFinal & isBearishPinBar

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(t3.iloc[i]) or pd.isna(hullmaHullMA.iloc[i]):
            continue

        entry_price = df['close'].iloc[i]

        if entryCondition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

        if entryConditionShort.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

    return entries