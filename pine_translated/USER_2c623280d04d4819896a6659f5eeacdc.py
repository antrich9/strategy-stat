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
    # Strategy parameters
    atrLength = 14
    atrMultiplier = 1.5
    lookback = 20
    retSince = 2
    retValid = 3
    bb = lookback

    # Repainting mode: 'On' (default)
    rTon = True
    rTcc = False
    rThv = False

    high = df['high']
    low = df['low']
    close = df['close']
    n = len(df)

    # Pivot points
    pl = pd.Series(np.nan, index=df.index)
    ph = pd.Series(np.nan, index=df.index)

    for i in range(bb, n - bb):
        pl.iloc[i] = low.iloc[i - bb:i + bb + 1].min()
        ph.iloc[i] = high.iloc[i - bb:i + bb + 1].max()

    # Box levels
    s_yLoc = pd.Series(np.nan, index=df.index)
    r_yLoc = pd.Series(np.nan, index=df.index)

    for i in range(bb + 1, n - bb - 1):
        if low.iloc[i + bb + 1] > low.iloc[i - bb - 1]:
            s_yLoc.iloc[i] = low.iloc[i - bb - 1]
        else:
            s_yLoc.iloc[i] = low.iloc[i + bb + 1]

        if high.iloc[i + bb + 1] > high.iloc[i - bb - 1]:
            r_yLoc.iloc[i] = high.iloc[i + bb + 1]
        else:
            r_yLoc.iloc[i] = high.iloc[i - bb - 1]

    # ATR (Wilder)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atrLength, adjust=False).mean()

    # Box tracking arrays
    sBox_top = pd.Series(np.nan, index=df.index)
    sBox_bot = pd.Series(np.nan, index=df.index)
    rBox_top = pd.Series(np.nan, index=df.index)
    rBox_bot = pd.Series(np.nan, index=df.index)

    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)
    sRetValid = pd.Series(False, index=df.index)
    rRetValid = pd.Series(False, index=df.index)

    # Box creation on pivot change
    for i in range(bb, n):
        if pd.notna(pl.iloc[i]) and (i == bb or pd.isna(pl.iloc[i-1])):
            sBox_top.iloc[i] = pl.iloc[i]
            sBox_bot.iloc[i] = s_yLoc.iloc[i]
        elif i > bb:
            sBox_top.iloc[i] = sBox_top.iloc[i-1]
            sBox_bot.iloc[i] = sBox_bot.iloc[i-1]

        if pd.notna(ph.iloc[i]) and (i == bb or pd.isna(ph.iloc[i-1])):
            rBox_top.iloc[i] = ph.iloc[i]
            rBox_bot.iloc[i] = r_yLoc.iloc[i]
        elif i > bb:
            rBox_top.iloc[i] = rBox_top.iloc[i-1]
            rBox_bot.iloc[i] = rBox_bot.iloc[i-1]

    # Detect box breakouts
    for i in range(bb + 1, n):
        sBot = sBox_bot.iloc[i-1] if pd.notna(sBox_bot.iloc[i-1]) else np.inf
        rTop = rBox_top.iloc[i-1] if pd.notna(rBox_top.iloc[i-1]) else -np.inf

        cu = close.iloc[i] < sBot and close.iloc[i-1] >= sBot
        co = close.iloc[i] > rTop and close.iloc[i-1] <= rTop

        if cu and not sBreak.iloc[i-1]:
            sBreak.iloc[i] = True
        if co and not rBreak.iloc[i-1]:
            rBreak.iloc[i] = True

        if pd.notna(pl.iloc[i]) and pd.isna(pl.iloc[i-1]) and not sBreak.iloc[i]:
            sBreak.iloc[i] = False
        if pd.notna(ph.iloc[i]) and pd.isna(ph.iloc[i-1]) and not rBreak.iloc[i]:
            rBreak.iloc[i] = False

        if i > bb:
            sBreak.iloc[i] = sBreak.iloc[i] or (sBreak.iloc[i-1] and pd.isna(pl.iloc[i]))
            rBreak.iloc[i] = rBreak.iloc[i] or (rBreak.iloc[i-1] and pd.isna(ph.iloc[i]))

    # Retest conditions for support
    for i in range(bb + retSince + 1, n):
        if sBreak.iloc[i]:
            bars_since_break = 1
            for j in range(i - 1, bb, -1):
                if not sBreak.iloc[j]:
                    bars_since_break = i - j
                    break

            if bars_since_break > retSince:
                sTop_val = sBox_top.iloc[i] if pd.notna(sBox_top.iloc[i]) else high.iloc[i]
                sBot_val = sBox_bot.iloc[i] if pd.notna(sBox_bot.iloc[i]) else low.iloc[i]

                if high.iloc[i] >= sTop_val and close.iloc[i] <= sBot_val:
                    sRetValid.iloc[i] = True
                elif high.iloc[i] >= sTop_val and close.iloc[i] >= sBot_val and close.iloc[i] <= sTop_val:
                    sRetValid.iloc[i] = True
                elif high.iloc[i] >= sBot_val and high.iloc[i] <= sTop_val:
                    sRetValid.iloc[i] = True
                elif high.iloc[i] >= sBot_val and high.iloc[i] <= sTop_val and close.iloc[i] < sBot_val:
                    sRetValid.iloc[i] = True

    # Retest conditions for resistance
    for i in range(bb + retSince + 1, n):
        if rBreak.iloc[i]:
            bars_since_break = 1
            for j in range(i - 1, bb, -1):
                if not rBreak.iloc[j]:
                    bars_since_break = i - j
                    break

            if bars_since_break > retSince:
                rTop_val = rBox_top.iloc[i] if pd.notna(rBox_top.iloc[i]) else high.iloc[i]
                rBot_val = rBox_bot.iloc[i] if pd.notna(rBox_bot.iloc[i]) else low.iloc[i]

                if low.iloc[i] <= rBot_val and close.iloc[i] >= rTop_val:
                    rRetValid.iloc[i] = True
                elif low.iloc[i] <= rBot_val and close.iloc[i] <= rTop_val and close.iloc[i] >= rBot_val:
                    rRetValid.iloc[i] = True
                elif low.iloc[i] <= rTop_val and low.iloc[i] >= rBot_val:
                    rRetValid.iloc[i] = True
                elif low.iloc[i] <= rTop_val and low.iloc[i] >= rBot_val and close.iloc[i] > rTop_val:
                    rRetValid.iloc[i] = True

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(bb + retSince + 1, n):
        if sRetValid.iloc[i]:
            entry_price = close.iloc[i]
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        if rRetValid.iloc[i]:
            entry_price = close.iloc[i]
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries