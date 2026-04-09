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

    asianSweptHigh = False
    asianSweptLow = False
    tempAsianHigh = np.nan
    tempAsianLow = np.nan
    asianHigh = np.nan
    asianLow = np.nan
    wasInSession = False

    pdHigh = np.nan
    pdLow = np.nan
    tempPdHigh = np.nan
    tempPdLow = np.nan
    pdSweptHigh = False
    pdSweptLow = False
    prev_date = None

    sweepMode = "None"
    trade_num = 1
    entries = []

    for i in range(len(df)):
        ts = int(df['time'].iloc[i])
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        date = dt.date()

        inAsianSession = dt.hour >= 19

        if inAsianSession and not wasInSession:
            tempAsianHigh = df['high'].iloc[i]
            tempAsianLow = df['low'].iloc[i]
        elif inAsianSession:
            if not np.isnan(tempAsianHigh):
                tempAsianHigh = max(tempAsianHigh, df['high'].iloc[i])
            else:
                tempAsianHigh = df['high'].iloc[i]
            if not np.isnan(tempAsianLow):
                tempAsianLow = min(tempAsianLow, df['low'].iloc[i])
            else:
                tempAsianLow = df['low'].iloc[i]

        if wasInSession and not inAsianSession:
            asianHigh = tempAsianHigh
            asianLow = tempAsianLow

        if inAsianSession and not wasInSession:
            if not asianSweptHigh and not np.isnan(asianHigh) and df['high'].iloc[i] > asianHigh:
                asianSweptHigh = True
            if not asianSweptLow and not np.isnan(asianLow) and df['low'].iloc[i] < asianLow:
                asianSweptLow = True

        if date != prev_date:
            if prev_date is not None:
                pdHigh = tempPdHigh
                pdLow = tempPdLow
                pdSweptHigh = False
                pdSweptLow = False
            tempPdHigh = df['high'].iloc[i]
            tempPdLow = df['low'].iloc[i]
        else:
            if not np.isnan(tempPdHigh):
                tempPdHigh = max(tempPdHigh, df['high'].iloc[i])
            else:
                tempPdHigh = df['high'].iloc[i]
            if not np.isnan(tempPdLow):
                tempPdLow = min(tempPdLow, df['low'].iloc[i])
            else:
                tempPdLow = df['low'].iloc[i]

        if not pdSweptHigh and not np.isnan(pdHigh) and df['high'].iloc[i] > pdHigh:
            pdSweptHigh = True
        if not pdSweptLow and not np.isnan(pdLow) and df['low'].iloc[i] < pdLow:
            pdSweptLow = True

        if i >= 2:
            c1High = df['high'].iloc[i-2]
            c1Low = df['low'].iloc[i-2]
            c2Close = df['close'].iloc[i-1]
            c2High = df['high'].iloc[i-1]
            c2Low = df['low'].iloc[i-1]

            if pd.isna(c1High) or pd.isna(c1Low) or pd.isna(c2Close):
                bullishBias = False
                bearishBias = False
            else:
                bullishBias = (c2Close > c1High) or (c2Low < c1Low and c2Close > c1Low)
                bearishBias = (c2Close < c1Low) or (c2High > c1High and c2Close < c2High)

            if not isinstance(bullishBias, bool):
                bullishBias = False
            if not isinstance(bearishBias, bool):
                bearishBias = False

            asianLongOk = asianSweptLow and not asianSweptHigh
            asianShortOk = asianSweptHigh and not asianSweptLow
            pdLongOk = pdSweptLow and not pdSweptHigh
            pdShortOk = pdSweptHigh and not pdSweptLow

            biasLongOk = bullishBias
            biasShortOk = bearishBias

            if sweepMode == "None":
                longSweepOk = True
                shortSweepOk = True
            elif sweepMode == "Asian Only":
                longSweepOk = asianLongOk
                shortSweepOk = asianShortOk
            elif sweepMode == "PD Only":
                longSweepOk = pdLongOk
                shortSweepOk = pdShortOk
            elif sweepMode == "Bias Only":
                longSweepOk = biasLongOk
                shortSweepOk = biasShortOk
            elif sweepMode == "Asian + Bias":
                longSweepOk = asianLongOk and biasLongOk
                shortSweepOk = asianShortOk and biasShortOk
            elif sweepMode == "PD + Bias":
                longSweepOk = pdLongOk and biasLongOk
                shortSweepOk = pdShortOk and biasShortOk
            elif sweepMode == "All Three":
                longSweepOk = asianLongOk and pdLongOk and biasLongOk
                shortSweepOk = asianShortOk and pdShortOk and biasShortOk
            else:
                longSweepOk = False
                shortSweepOk = False

            if bullishBias and longSweepOk:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': df['close'].iloc[i],
                    'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
                    'raw_price_a': df['close'].iloc[i], 'raw_price_b': df['close'].iloc[i]})
                trade_num += 1
            elif bearishBias and shortSweepOk:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': df['close'].iloc[i],
                    'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
                    'raw_price_a': df['close'].iloc[i], 'raw_price_b': df['close'].iloc[i]})
                trade_num += 1

        wasInSession = inAsianSession
        prev_date = date

    return entries