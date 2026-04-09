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
    # Parameters
    swingLen = 14
    confirmBars = 3
    atrFrac = 0.5
    fvgWaitBars = 10
    fvgMinTicks = 3
    atrLength = 14
    useSession = False
    sessionTime = "0930-1600"
    useFVG = True

    n = len(df)
    entries = []
    trade_num = 1

    # Session filter
    if useSession:
        session_mask = pd.Series([False] * n, index=df.index)
        for i in range(n):
            dt = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc)
            time_str = dt.strftime("%H%M")
            date_str = dt.strftime("%Y-%m-%d")
            start_str = sessionTime.split("-")[0]
            end_str = sessionTime.split("-")[1]
            start_min = int(start_str[:2]) * 60 + int(start_str[2:])
            end_min = int(end_str[:2]) * 60 + int(end_str[2:])
            curr_min = int(time_str[:2]) * 60 + int(time_str[2:])
            session_mask.iloc[i] = (curr_min >= start_min and curr_min < end_min)
    else:
        session_mask = pd.Series([True] * n, index=df.index)

    # Calculate ATR (Wilder)
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift(1))
    tr3 = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=atrLength, adjust=False).mean()
    buf = atr * atrFrac

    # Pivot High/Low
    pivothigh = pd.Series(np.nan, index=df.index)
    pivotlow = pd.Series(np.nan, index=df.index)

    for i in range(swingLen, n - swingLen):
        window = df['high'].iloc[i - swingLen:i + swingLen + 1]
        if df['high'].iloc[i] == window.max():
            pivothigh.iloc[i] = df['high'].iloc[i]
        window = df['low'].iloc[i - swingLen:i + swingLen + 1]
        if df['low'].iloc[i] == window.min():
            pivotlow.iloc[i] = df['low'].iloc[i]

    # FVG Detection (vectorized)
    high_2 = df['high'].shift(2)
    low_2 = df['low'].shift(2)
    bullish_fvg = df['low'] > high_2
    bullish_fvg_size = np.where(bullish_fvg, df['low'] - high_2, 0.0)
    bullish_fvg_valid = bullish_fvg & (bullish_fvg_size >= fvgMinTicks)

    bearish_fvg = df['high'] < low_2
    bearish_fvg_size = np.where(bearish_fvg, low_2 - df['high'], 0.0)
    bearish_fvg_valid = bearish_fvg & (bearish_fvg_size >= fvgMinTicks)

    # State variables
    lastHi = np.nan
    lastLo = np.nan
    lastBearGrabBar = np.nan
    lastBullGrabBar = np.nan

    pendHi = False
    pendHiLevel = np.nan
    pendHiBars = 0
    pendHiExtremeY = np.nan

    pendLo = False
    pendLoLevel = np.nan
    pendLoBars = 0
    pendLoExtremeY = np.nan

    # Iterate bars
    for i in range(n):
        if pd.isna(df['high'].iloc[i]) or pd.isna(df['close'].iloc[i]):
            continue

        # Update swing levels from pivots
        if not pd.isna(pivothigh.iloc[i]):
            lastHi = pivothigh.iloc[i]
        if not pd.isna(pivotlow.iloc[i]):
            lastLo = pivotlow.iloc[i]

        # Skip bars with NaN lastHi/Lo or ATR
        if pd.isna(lastHi) or pd.isna(lastLo) or pd.isna(atr.iloc[i]):
            continue

        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        current_close = df['close'].iloc[i]
        prev_high = df['high'].iloc[i - 1] if i > 0 else current_high
        prev_low = df['low'].iloc[i - 1] if i > 0 else current_low
        prev_close = df['close'].iloc[i - 1] if i > 0 else current_close

        buf_val = buf.iloc[i]
        lastHi_val = lastHi
        lastLo_val = lastLo

        # Take conditions
        takeAbove = current_high > lastHi_val + buf_val
        takeBelow = current_low < lastLo_val - buf_val

        firstBreachAbove = takeAbove and (prev_high <= lastHi_val + buf_val)
        firstBreachBelow = takeBelow and (prev_low >= lastLo_val - buf_val)

        # Same candle grabs
        sameCandleBearGrab = firstBreachAbove and (current_close < lastHi_val)
        sameCandleBullGrab = firstBreachBelow and (current_close > lastLo_val)

        # Start pending
        if (not pendHi) and firstBreachAbove and (current_close >= lastHi_val):
            pendHi = True
            pendHiLevel = lastHi_val
            pendHiBars = 0
            pendHiExtremeY = current_high

        if (not pendLo) and firstBreachBelow and (current_close <= lastLo_val):
            pendLo = True
            pendLoLevel = lastLo_val
            pendLoBars = 0
            pendLoExtremeY = current_low

        # Multi-candle grab tracking
        bearGrab = sameCandleBearGrab
        bullGrab = sameCandleBullGrab

        if pendHi:
            if np.isnan(pendHiExtremeY) or current_high > pendHiExtremeY:
                pendHiExtremeY = current_high

            if current_close < pendHiLevel:
                bearGrab = True
                lastBearGrabBar = i
                pendHi = False
                pendHiLevel = np.nan
                pendHiBars = 0
                pendHiExtremeY = np.nan
                lastHi = np.nan
            else:
                pendHiBars += 1
                if pendHiBars >= confirmBars:
                    pendHi = False
                    pendHiLevel = np.nan
                    pendHiBars = 0
                    pendHiExtremeY = np.nan

        if pendLo:
            if np.isnan(pendLoExtremeY) or current_low < pendLoExtremeY:
                pendLoExtremeY = current_low

            if current_close > pendLoLevel:
                bullGrab = True
                lastBullGrabBar = i
                pendLo = False
                pendLoLevel = np.nan
                pendLoBars = 0
                pendLoExtremeY = np.nan
                lastLo = np.nan
            else:
                pendLoBars += 1
                if pendLoBars >= confirmBars:
                    pendLo = False
                    pendLoLevel = np.nan
                    pendLoBars = 0
                    pendLoExtremeY = np.nan

        # Clear old grabs
        if not pd.isna(lastBearGrabBar) and (i - lastBearGrabBar) > fvgWaitBars:
            lastBearGrabBar = np.nan
        if not pd.isna(lastBullGrabBar) and (i - lastBullGrabBar) > fvgWaitBars:
            lastBullGrabBar = np.nan

        # Bars since grabs
        barsSinceBearGrab = 999 if pd.isna(lastBearGrabBar) else (i - lastBearGrabBar)
        barsSinceBullGrab = 999 if pd.isna(lastBullGrabBar) else (i - lastBullGrabBar)

        # Long Entry
        longSetup = bullGrab or (barsSinceBullGrab > 0 and barsSinceBullGrab <= fvgWaitBars)
        longEntry = longSetup and bullish_fvg_valid.iloc[i] and session_mask.iloc[i]

        if longEntry:
            if useFVG:
                if barsSinceBullGrab > 0:
                    lastBullGrabBar = np.nan
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            lastBullGrabBar = np.nan
            bullGrab = False

        # Short Entry
        shortSetup = bearGrab or (barsSinceBearGrab > 0 and barsSinceBearGrab <= fvgWaitBars)
        shortEntry = shortSetup and bearish_fvg_valid.iloc[i] and session_mask.iloc[i]

        if shortEntry:
            if useFVG:
                if barsSinceBearGrab > 0:
                    lastBearGrabBar = np.nan
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            lastBearGrabBar = np.nan
            bearGrab = False

    return entries