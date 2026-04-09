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
    leftBars = 10
    rightBars = leftBars - 2
    volFilter = "Mid"
    dynamic = False
    fvgMinTicks = 3
    waitForFVG = True
    fvgWaitBars = 10

    filter_val = {"Low": 1, "Mid": 2, "High": 3}[volFilter]

    high = df['high']
    low = df['low']
    close = df['close']
    volume_df = df['volume']

    avg_vol = volume_df.rolling(rightBars).mean()
    normalized_vol = avg_vol / avg_vol.rolling(500).std()

    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    aTR = tr.ewm(alpha=1.0/200, adjust=False).mean()

    isUpperGrabbed = pd.Series(False, index=df.index)
    isLowerGrabbed = pd.Series(False, index=df.index)

    for i in range(1, len(df) - rightBars):
        if i < leftBars:
            continue

        left_start = i - leftBars
        left_end = i - rightBars - 1

        ph_candidate = high.iloc[i]
        left_max = high.iloc[left_start:left_end+1].max()
        right_end_idx = i + rightBars
        right_max = high.iloc[i-rightBars+1:right_end_idx+1].max()

        pl_candidate = low.iloc[i]
        left_min = low.iloc[left_start:left_end+1].min()
        right_min = low.iloc[i-rightBars+1:right_end_idx+1].min()

        if ph_candidate > left_max and ph_candidate > right_max:
            if normalized_vol.iloc[i] > filter_val:
                isUpperGrabbed.iloc[i] = True

        if pl_candidate < left_min and pl_candidate < right_min:
            if normalized_vol.iloc[i] > filter_val:
                isLowerGrabbed.iloc[i] = True

    entries = []
    trade_num = 1
    lastUpperGrabBar = None
    lastLowerGrabBar = None

    for i in range(1, len(df) - rightBars):
        if isUpperGrabbed.iloc[i]:
            lastUpperGrabBar = i

        if isLowerGrabbed.iloc[i]:
            lastLowerGrabBar = i

        barsSinceUpperGrab = 999 if lastUpperGrabBar is None else (i - lastUpperGrabBar)
        barsSinceLowerGrab = 999 if lastLowerGrabBar is None else (i - lastLowerGrabBar)

        if barsSinceUpperGrab > fvgWaitBars:
            lastUpperGrabBar = None
        if barsSinceLowerGrab > fvgWaitBars:
            lastLowerGrabBar = None

        bullishFVG = low.iloc[i] > high.iloc[i-2]
        bullishFVGSize = (low.iloc[i] - high.iloc[i-2]) if bullishFVG else 0
        bullishFVGValid = bullishFVG and (bullishFVGSize >= fvgMinTicks * 1)

        bearishFVG = high.iloc[i] < low.iloc[i-2]
        bearishFVGSize = (low.iloc[i-2] - high.iloc[i]) if bearishFVG else 0
        bearishFVGValid = bearishFVG and (bearishFVGSize >= fvgMinTicks * 1)

        shortEntry = (barsSinceUpperGrab > 0 and barsSinceUpperGrab <= fvgWaitBars) and (not waitForFVG or bearishFVGValid)
        longEntry = (barsSinceLowerGrab > 0 and barsSinceLowerGrab <= fvgWaitBars) and (not waitForFVG or bullishFVGValid)

        if longEntry:
            lastLowerGrabBar = None
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1

        if shortEntry:
            lastUpperGrabBar = None
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1

    return entries