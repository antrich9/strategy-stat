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
    df = df.copy().reset_index(drop=True)
    n = len(df)

    # Session detection helper
    def is_in_session(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        current_mins = hour * 60 + minute
        start_mins = 23 * 60  # 2300
        end_mins = 7 * 60    # 0700
        if start_mins <= end_mins:
            return start_mins <= current_mins < end_mins
        else:
            return current_mins >= start_mins or current_mins < end_mins

    inSession = df['time'].apply(is_in_session)
    inSession_prev = inSession.shift(1).fillna(False).astype(bool)
    newSession = inSession & ~inSession_prev
    sessionEnd = ~inSession & inSession_prev

    # Track Asian session high/low
    asiaHigh = pd.Series(np.nan, index=df.index)
    asiaLow = pd.Series(np.nan, index=df.index)

    current_asiaHigh = np.nan
    current_asiaLow = np.nan

    for i in range(n):
        if newSession.iloc[i]:
            current_asiaHigh = df['high'].iloc[i]
            current_asiaLow = df['low'].iloc[i]
        elif inSession.iloc[i]:
            if not np.isnan(current_asiaHigh):
                current_asiaHigh = max(current_asiaHigh, df['high'].iloc[i])
            if not np.isnan(current_asiaLow):
                current_asiaLow = min(current_asiaLow, df['low'].iloc[i])
        asiaHigh.iloc[i] = current_asiaHigh
        asiaLow.iloc[i] = current_asiaLow

    asiaHighPlot = pd.Series(np.nan, index=df.index)
    asiaLowPlot = pd.Series(np.nan, index=df.index)
    prev_asiaHighPlot = np.nan
    prev_asiaLowPlot = np.nan

    for i in range(n):
        if sessionEnd.iloc[i]:
            prev_asiaHighPlot = asiaHigh.iloc[i]
            prev_asiaLowPlot = asiaLow.iloc[i]
            asiaHighPlot.iloc[i] = asiaHigh.iloc[i]
            asiaLowPlot.iloc[i] = asiaLow.iloc[i]
        elif not inSession.iloc[i]:
            asiaHighPlot.iloc[i] = prev_asiaHighPlot
            asiaLowPlot.iloc[i] = prev_asiaLowPlot

    asiahighSwept = df['high'] > asiaHighPlot
    asialowSwept = df['low'] < asiaLowPlot

    # OB conditions
    isUp = df['close'] > df['open']
    isDown = df['close'] < df['open']

    def isObUp(idx):
        i = idx
        if i < 1 or i + 1 >= n:
            return False
        cond1 = isDown.iloc[i + 1]
        cond2 = isUp.iloc[i]
        cond3 = df['close'].iloc[i] > df['high'].iloc[i + 1]
        return cond1 and cond2 and cond3

    def isObDown(idx):
        i = idx
        if i < 1 or i + 1 >= n:
            return False
        cond1 = isUp.iloc[i + 1]
        cond2 = isDown.iloc[i]
        cond3 = df['close'].iloc[i] < df['low'].iloc[i + 1]
        return cond1 and cond2 and cond3

    obUp = pd.Series([isObUp(i) for i in range(n)], index=df.index)
    obDown = pd.Series([isObDown(i) for i in range(n)], index=df.index)

    fvgUp = df['low'] > df['high'].shift(2)
    fvgDown = df['high'] < df['low'].shift(2)

    # Filters
    volfilt = df['volume'] > df['volume'].rolling(9).mean() * 1.5

    # Wilder ATR
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': (df['high'] - df['close'].shift(1)).abs(),
        'lc': (df['low'] - df['close'].shift(1)).abs()
    }).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()

    atrfilt = (df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr)

    loc = df['close'].ewm(span=54, adjust=False).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    bfvg = df['low'] > df['high'].shift(2) & volfilt & atrfilt & locfiltb
    sfvg = df['high'] < df['low'].shift(2) & volfilt & atrfilt & locfilts

    # Entry conditions
    long_cond = asiahighSwept & obUp & fvgUp & volfilt & atrfilt & locfiltb
    short_cond = asialowSwept & obDown & fvgDown & volfilt & atrfilt & locfilts

    entries = []
    trade_num = 1

    for i in range(2, n):
        if newSession.iloc[i] or sessionEnd.iloc[i]:
            continue
        if asiaHighPlot.iloc[i] != asiaHighPlot.iloc[i] or asiaLowPlot.iloc[i] != asiaLowPlot.iloc[i]:
            continue

        entry_price_guess = df['close'].iloc[i]
        ts = int(df['time'].iloc[i])
        t_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        if long_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': t_str,
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1

        if short_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': t_str,
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1

    return entries