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
    # Default parameters from script
    bb = 20
    input_retSince = 2
    input_retValid = 2
    tradeDirection = "Both"

    results = []
    trade_num = 0

    n = len(df)
    if n < bb + 1:
        return results

    # Calculate pivot low and pivot high
    pl = pd.Series(np.nan, index=df.index)
    ph = pd.Series(np.nan, index=df.index)

    for i in range(bb, n - bb):
        pl_val = df['low'].iloc[i - bb:i + bb + 1].min()
        if df['low'].iloc[i] == pl_val:
            pl.iloc[i] = df['low'].iloc[i]

        ph_val = df['high'].iloc[i - bb:i + bb + 1].max()
        if df['high'].iloc[i] == ph_val:
            ph.iloc[i] = df['high'].iloc[i]

    # Support box
    sBreak = pd.Series(False, index=df.index)
    sBot_vals = pd.Series(np.nan, index=df.index)
    sTop_vals = pd.Series(np.nan, index=df.index)
    sBox_bottom = pd.Series(np.nan, index=df.index)
    sBox_top = pd.Series(np.nan, index=df.index)

    # Resistance box
    rBreak = pd.Series(False, index=df.index)
    rBot_vals = pd.Series(np.nan, index=df.index)
    rTop_vals = pd.Series(np.nan, index=df.index)
    rBox_bottom = pd.Series(np.nan, index=df.index)
    rBox_top = pd.Series(np.nan, index=df.index)

    # Calculate support and resistance box levels
    for i in range(bb, n):
        if pd.notna(pl.iloc[i]) and pd.isna(pl.iloc[i - 1]):
            s_yLoc = df['low'].iloc[i + bb + 1] if i + bb + 1 < n else df['low'].iloc[i + bb]
            if df['low'].iloc[i - bb - 1] > df['low'].iloc[i + bb]:
                s_yLoc = df['low'].iloc[i + bb]
            sBox_bottom.iloc[i] = pl.iloc[i]
            sBox_top.iloc[i] = pl.iloc[i]
            if df['low'].iloc[i + bb + 1] > df['low'].iloc[i - bb - 1]:
                s_yLoc = df['low'].iloc[i - bb - 1]
            sBot_vals.iloc[i] = s_yLoc
            sTop_vals.iloc[i] = pl.iloc[i]

        if pd.notna(ph.iloc[i]) and pd.isna(ph.iloc[i - 1]):
            r_yLoc = df['high'].iloc[i + bb + 1] if i + bb + 1 < n else df['high'].iloc[i + bb]
            if df['high'].iloc[i - bb - 1] > df['high'].iloc[i + bb]:
                r_yLoc = df['high'].iloc[i - bb - 1]
            rBot_vals.iloc[i] = ph.iloc[i]
            rTop_vals.iloc[i] = ph.iloc[i]
            if df['high'].iloc[i + bb + 1] > df['high'].iloc[i - bb - 1]:
                r_yLoc = df['high'].iloc[i + bb + 1]
            rBot_vals.iloc[i] = ph.iloc[i]
            rTop_vals.iloc[i] = r_yLoc

    # Fill forward box values
    sBot_filled = sBot_vals.ffill()
    sTop_filled = sTop_vals.ffill()
    rBot_filled = rBot_vals.ffill()
    rTop_filled = rTop_vals.ffill()

    # Calculate breakout conditions
    cu = pd.Series(False, index=df.index)
    co = pd.Series(False, index=df.index)

    for i in range(1, n):
        if pd.notna(sBot_filled.iloc[i]) and pd.notna(sBot_filled.iloc[i - 1]):
            if df['close'].iloc[i] < sBot_filled.iloc[i] and df['close'].iloc[i - 1] >= sBot_filled.iloc[i - 1]:
                cu.iloc[i] = True

        if pd.notna(rTop_filled.iloc[i]) and pd.notna(rTop_filled.iloc[i - 1]):
            if df['close'].iloc[i] > rTop_filled.iloc[i] and df['close'].iloc[i - 1] <= rTop_filled.iloc[i - 1]:
                co.iloc[i] = True

    # Calculate retest conditions for support
    sRetValid = pd.Series(False, index=df.index)
    sBreak_state = False
    sBreak_idx = -1

    for i in range(bb + 1, n):
        if cu.iloc[i] and not sBreak_state:
            sBreak_state = True
            sBreak_idx = i

        if pd.notna(pl.iloc[i]) and pd.notna(pl.iloc[i - 1]):
            if pd.isna(sBreak_state) or not sBreak_state:
                pass
            sBreak_state = False
            sBreak_idx = -1

        if sBreak_state and sBreak_idx >= 0:
            bars_since_break = i - sBreak_idx
            if bars_since_break > input_retSince:
                sTop_val = sTop_filled.iloc[sBreak_idx] if pd.notna(sTop_filled.iloc[sBreak_idx]) else df['high'].iloc[sBreak_idx]
                sBot_val = sBot_filled.iloc[sBreak_idx] if pd.notna(sBot_filled.iloc[sBreak_idx]) else df['low'].iloc[sBreak_idx]

                if pd.notna(sTop_val) and pd.notna(sBot_val):
                    c1 = df['high'].iloc[i] >= sTop_val and df['close'].iloc[i] <= sBot_val
                    c2 = df['high'].iloc[i] >= sTop_val and df['close'].iloc[i] >= sBot_val and df['close'].iloc[i] <= sTop_val
                    c3 = df['high'].iloc[i] >= sBot_val and df['high'].iloc[i] <= sTop_val
                    c4 = df['high'].iloc[i] >= sBot_val and df['high'].iloc[i] <= sTop_val and df['close'].iloc[i] < sBot_val

                    if c1 or c2 or c3 or c4:
                        if bars_since_break > 0 and bars_since_break <= input_retValid:
                            sRetValid.iloc[i] = True

    # Calculate retest conditions for resistance
    rRetValid = pd.Series(False, index=df.index)
    rBreak_state = False
    rBreak_idx = -1

    for i in range(bb + 1, n):
        if co.iloc[i] and not rBreak_state:
            rBreak_state = True
            rBreak_idx = i

        if pd.notna(ph.iloc[i]) and pd.notna(ph.iloc[i - 1]):
            if pd.isna(rBreak_state) or not rBreak_state:
                pass
            rBreak_state = False
            rBreak_idx = -1

        if rBreak_state and rBreak_idx >= 0:
            bars_since_break = i - rBreak_idx
            if bars_since_break > input_retSince:
                rTop_val = rTop_filled.iloc[rBreak_idx] if pd.notna(rTop_filled.iloc[rBreak_idx]) else df['high'].iloc[rBreak_idx]
                rBot_val = rBot_filled.iloc[rBreak_idx] if pd.notna(rBot_filled.iloc[rBreak_idx]) else df['low'].iloc[rBreak_idx]

                if pd.notna(rTop_val) and pd.notna(rBot_val):
                    c1 = df['low'].iloc[i] <= rBot_val and df['close'].iloc[i] >= rTop_val
                    c2 = df['low'].iloc[i] <= rBot_val and df['close'].iloc[i] <= rTop_val and df['close'].iloc[i] >= rBot_val
                    c3 = df['low'].iloc[i] <= rTop_val and df['low'].iloc[i] >= rBot_val
                    c4 = df['low'].iloc[i] <= rTop_val and df['low'].iloc[i] >= rBot_val and df['close'].iloc[i] > rTop_val

                    if c1 or c2 or c3 or c4:
                        if bars_since_break > 0 and bars_since_break <= input_retValid:
                            rRetValid.iloc[i] = True

    # Generate entries based on trade direction
    for i in range(bb + 1, n):
        if pd.isna(df['close'].iloc[i]):
            continue

        direction = None

        if tradeDirection == "Both":
            if co.iloc[i]:
                direction = 'long'
            elif cu.iloc[i]:
                direction = 'short'
        elif tradeDirection == "Long":
            if co.iloc[i]:
                direction = 'long'
        elif tradeDirection == "Short":
            if cu.iloc[i]:
                direction = 'short'

        if direction is not None:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])

            results.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

    return results