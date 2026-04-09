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

    # Parameters from script
    length_turtle = 20
    len2_turtle = 10

    # Turtle definitions
    upper_turtle = high.rolling(length_turtle).max().shift(1)
    lower_turtle = low.rolling(length_turtle).min().shift(1)

    up_turtle = high.rolling(length_turtle).max()
    down_turtle = low.rolling(length_turtle).min()
    sup_turtle = high.rolling(len2_turtle).max()
    sdown_turtle = low.rolling(len2_turtle).min()

    K1_turtle_vals = np.where(
        (high >= up_turtle.shift(1)).rolling(length_turtle).sum() <= (low <= down_turtle.shift(1)).rolling(length_turtle).sum(),
        down_turtle, up_turtle
    )
    K1_turtle = pd.Series(K1_turtle_vals, index=high.index)
    K2_turtle_vals = np.where(
        (high >= up_turtle.shift(1)).rolling(length_turtle).sum() <= (low <= down_turtle.shift(1)).rolling(len2_turtle).sum(),
        sdown_turtle, sup_turtle
    )
    K2_turtle = pd.Series(K2_turtle_vals, index=high.index)

    # Turtle signals
    buySignal_turtle = (high == upper_turtle) | ((high > upper_turtle) & (high.shift(1) <= upper_turtle.shift(1)))
    sellSignal_turtle = (low == lower_turtle) | ((low < lower_turtle) & (low.shift(1) >= lower_turtle.shift(1)))
    buyExit_turtle = (low == sdown_turtle.shift(1)) | ((low < sdown_turtle.shift(1)) & (low.shift(1) >= sdown_turtle.shift(1)))
    sellExit_turtle = (high == sup_turtle.shift(1)) | ((high > sup_turtle.shift(1)) & (high.shift(1) <= sup_turtle.shift(1)))

    # barssince implementations
    def barssince(cond):
        result = pd.Series(np.nan, index=cond.index)
        count = -1
        for i in range(len(cond)):
            if pd.notna(cond.iloc[i]) and cond.iloc[i]:
                count = 0
            elif count >= 0:
                count += 1
            if count >= 0:
                result.iloc[i] = count
        return result

    O1_turtle = barssince(buySignal_turtle)
    O2_turtle = barssince(sellSignal_turtle)
    O3_turtle = barssince(buyExit_turtle)
    O4_turtle = barssince(sellExit_turtle)

    # Turtle basic conditions
    basicLongConditionTurtle = buySignal_turtle & (O3_turtle < O1_turtle.shift(1))
    basicShortConditionTurtle = sellSignal_turtle & (O4_turtle < O2_turtle.shift(1))

    TurtleSignalsLong = basicLongConditionTurtle
    TurtleSignalsShort = basicShortConditionTurtle

    # Cross confirmation
    TurtleSignalsLongCross = (~TurtleSignalsLong.shift(1).fillna(False)) & TurtleSignalsLong
    TurtleSignalsShortCross = (~TurtleSignalsShort.shift(1).fillna(False)) & TurtleSignalsShort

    # TDFI calculation
    lookbackTDFI = 13
    mmaLengthTDFI = 13
    smmaLengthTDFI = 13
    nLengthTDFI = 3
    filterHighTDFI = 0.05
    filterLowTDFI = -0.05
    priceTDFI = close

    # TEMA
    def tema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3

    # EMA function
    def ma_func(src, length):
        return src.ewm(span=length, adjust=False).mean()

    # TDFI main calculation
    mmaTDFI = ma_func(priceTDFI * 1000, mmaLengthTDFI)
    smmaTDFI = ma_func(mmaTDFI, smmaLengthTDFI)
    impetmmaTDFI = mmaTDFI - mmaTDFI.shift(1)
    impetsmmaTDFI = smmaTDFI - smmaTDFI.shift(1)
    divmaTDFI = (mmaTDFI - smmaTDFI).abs()
    averimpetTDFI = (impetmmaTDFI + impetsmmaTDFI) / 2
    tdfTDFI_raw = (divmaTDFI ** 1) * (averimpetTDFI ** nLengthTDFI)
    highest_tdf = tdfTDFI_raw.abs().rolling(lookbackTDFI * nLengthTDFI).max()
    signalTDFI = tdfTDFI_raw / highest_tdf

    # TDFI signals
    crossTDFI = True
    inverseTDFI = True

    signalLongTDFI = signalTDFI > filterHighTDFI
    signalShortTDFI = signalTDFI < filterLowTDFI

    finalLongSignalTDFI = signalShortTDFI  # inverseTDFI
    finalShortSignalTDFI = signalLongTDFI  # inverseTDFI

    # Wilder ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr1 = tr.ewm(alpha=1/14, adjust=False).mean()

    # Combined entry conditions
    entryLongCondition = TurtleSignalsLongCross
    entryShortCondition = TurtleSignalsShortCross

    entries = []
    trade_num = 1

    # Iterate through bars
    for i in range(2, len(df)):
        if pd.isna(upper_turtle.iloc[i]) or pd.isna(lower_turtle.iloc[i]) or pd.isna(atr1.iloc[i]) or pd.isna(signalTDFI.iloc[i]):
            continue

        if entryLongCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

        if entryShortCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

    return entries