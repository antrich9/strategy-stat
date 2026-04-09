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
    df = df.copy()
    df['ny_hour'] = pd.to_datetime(df['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.hour
    df['utc_hour'] = pd.to_datetime(df['time'], unit='ms').dt.tz_localize('UTC').dt.hour
    df['new_day'] = df['time'].diff().dt.days != 0
    df['new_day'].fillna(False, inplace=True)

    inAsianSession = df['ny_hour'] >= 19
    sessionJustStarted = inAsianSession & ~inAsianSession.shift(1).fillna(False)
    asianSessionEnded = ~inAsianSession & inAsianSession.shift(1).fillna(False)

    asianHigh = pd.Series(np.nan, index=df.index)
    asianLow = pd.Series(np.nan, index=df.index)
    tempAsianHigh = pd.Series(np.nan, index=df.index)
    tempAsianLow = pd.Series(np.nan, index=df.index)

    tempAsianHigh_val = np.nan
    tempAsianLow_val = np.nan
    asianHigh_val = np.nan
    asianLow_val = np.nan
    sweptHigh = False
    sweptLow = False
    bothSwept = False

    for i in range(len(df)):
        if sessionJustStarted.iloc[i]:
            tempAsianHigh_val = df['high'].iloc[i]
            tempAsianLow_val = df['low'].iloc[i]
            sweptHigh = False
            sweptLow = False
            bothSwept = False
        elif inAsianSession.iloc[i]:
            tempAsianHigh_val = max(tempAsianHigh_val, df['high'].iloc[i])
            tempAsianLow_val = min(tempAsianLow_val, df['low'].iloc[i])
        if asianSessionEnded.iloc[i]:
            asianHigh_val = tempAsianHigh_val
            asianLow_val = tempAsianLow_val
        asianHigh.iloc[i] = asianHigh_val
        asianLow.iloc[i] = asianLow_val

        high = df['high'].iloc[i]
        low = df['low'].iloc[i]

        if not sweptHigh and not np.isnan(asianHigh_val) and high > asianHigh_val:
            sweptHigh = True
        if not sweptLow and not np.isnan(asianLow_val) and low < asianLow_val:
            sweptLow = True
        if sweptHigh and sweptLow and not bothSwept:
            bothSwept = True

    newDay = df['new_day']
    pdHigh = pd.Series(np.nan, index=df.index)
    pdLow = pd.Series(np.nan, index=df.index)
    tempHigh1_val = np.nan
    tempLow1_val = np.nan
    pdHigh_val = np.nan
    pdLow_val = np.nan
    pdSweptHigh = False
    pdSweptLow = False

    for i in range(len(df)):
        if newDay.iloc[i]:
            pdHigh_val = tempHigh1_val
            pdLow_val = tempLow1_val
            tempHigh1_val = np.nan
            tempLow1_val = np.nan
            pdSweptHigh = False
            pdSweptLow = False
        pdHigh.iloc[i] = pdHigh_val
        pdLow.iloc[i] = pdLow_val

        high = df['high'].iloc[i]
        low = df['low'].iloc[i]

        if np.isnan(tempHigh1_val):
            tempHigh1_val = high
        else:
            tempHigh1_val = max(tempHigh1_val, high)
        if np.isnan(tempLow1_val):
            tempLow1_val = low
        else:
            tempLow1_val = min(tempLow1_val, low)

        if not pdSweptHigh and not np.isnan(pdHigh_val) and high > pdHigh_val:
            pdSweptHigh = True
        if not pdSweptLow and not np.isnan(pdLow_val) and low < pdLow_val:
            pdSweptLow = True

    isWithinMorningWindow = (df['utc_hour'] >= 8) & (df['utc_hour'] < 9) | (df['utc_hour'] == 9) & (df['utc_hour'] < 10)
    isWithinAfternoonWindow1 = (df['utc_hour'] >= 10) & (df['utc_hour'] < 11)
    in_trading_window = isWithinMorningWindow | isWithinAfternoonWindow1

    opens = df['open']
    highs = df['high']
    lows = df['low']
    closes = df['close']

    isUp2 = closes.shift(2) > opens.shift(2)
    isDown2 = closes.shift(2) < opens.shift(2)
    isUp1 = closes.shift(1) > opens.shift(1)
    isDown1 = closes.shift(1) < opens.shift(1)

    obUp = isDown2 & isUp1 & (closes.shift(1) > highs.shift(2))
    obDown = isUp2 & isDown1 & (closes.shift(1) < lows.shift(2))

    fvgUp = lows > highs.shift(2)
    fvgDown = highs < lows.shift(2)

    stackedUp = obUp & fvgUp
    stackedDown = obDown & fvgDown

    inverseFvgUp = (highs.shift(2) < lows) & (closes.shift(1) >= highs.shift(2))
    inverseFvgDown = (lows.shift(2) > highs) & (closes.shift(1) <= lows.shift(2))

    sharpTurnUp = fvgUp.shift(1) & fvgDown
    sharpTurnDown = fvgDown.shift(1) & fvgUp

    doubleFvgUp = fvgUp & fvgUp.shift(1)
    doubleFvgDown = fvgDown & fvgDown.shift(1)

    bfvgUp = fvgUp & (lows <= highs.shift(1)) & (highs >= lows.shift(1))
    bfvgDown = fvgDown & (lows.shift(1) <= highs) & (highs.shift(1) >= lows)

    engulfingUp = isDown2 & isUp1 & (closes.shift(1) > highs.shift(2))
    engulfingDown = isUp2 & isDown1 & (closes.shift(1) < lows.shift(2))

    useObOnly = False
    useObFvgStacked = True
    useFvgOnly = False
    useInverseFvg = False
    useSharpTurn = False
    useDoubleFvg = False
    useBfvg = False
    useEngulfing = False

    sweepMode = "Both"

    patternLong = pd.Series(False, index=df.index)
    patternShort = pd.Series(False, index=df.index)

    if useObOnly:
        patternLong = patternLong | obUp
        patternShort = patternShort | obDown
    if useObFvgStacked:
        patternLong = patternLong | stackedUp
        patternShort = patternShort | stackedDown
    if useFvgOnly:
        patternLong = patternLong | fvgUp
        patternShort = patternShort | fvgDown
    if useInverseFvg:
        patternLong = patternLong | inverseFvg