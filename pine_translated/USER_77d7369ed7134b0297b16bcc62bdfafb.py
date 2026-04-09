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
    bb = 20  # input_lookback
    input_retSince = 2  # Bars Since Breakout
    input_retValid = 2  # Retest Detection Limiter
    input_repType = 'On'  # Repainting mode (default 'On')
    atrLength = 14  # ATR Length

    # Helper: Wilder RSI/ATR smoothing
    def wilder_smooth(series, period):
        alpha = 1.0 / period
        result = pd.Series(index=series.index, dtype=float)
        result.iloc[0] = series.iloc[0]
        for i in range(1, len(series)):
            result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
        return result

    # Calculate ATR (Wilder ATR)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = wilder_smooth(true_range, atrLength)

    # Calculate pivot points
    pl = pd.Series(np.nan, index=df.index)  # pivotlow
    ph = pd.Series(np.nan, index=df.index)  # pivothigh

    for i in range(bb, len(df) - bb):
        # pivotlow: lowest low over bb bars ending at i-bb
        window_low = df['low'].iloc[i-bb:i+1]
        idx_min = window_low.idxmin()
        # Check if this bar (i-bb) is the lowest
        if df['low'].iloc[i-bb] == window_low.min():
            pl.iloc[i] = df['low'].iloc[i-bb]

        # pivothigh: highest high over bb bars ending at i-bb
        window_high = df['high'].iloc[i-bb:i+1]
        idx_max = window_high.idxmax()
        if df['high'].iloc[i-bb] == window_high.max():
            ph.iloc[i] = df['high'].iloc[i-bb]

    # Fill NaN with previous values using fixnan behavior
    pl = pl.ffill()
    ph = ph.ffill()

    # Calculate box levels
    s_yLoc = pd.Series(np.nan, index=df.index)
    r_yLoc = pd.Series(np.nan, index=df.index)
    sBot = pd.Series(np.nan, index=df.index)  # support box bottom
    sTop = pd.Series(np.nan, index=df.index)  # support box top
    rBot = pd.Series(np.nan, index=df.index)  # resistance box bottom
    rTop = pd.Series(np.nan, index=df.index)  # resistance box top

    for i in range(bb + 1, len(df)):
        s_yLoc.iloc[i] = df['low'].iloc[bb - 1] if df['low'].iloc[bb + 1] > df['low'].iloc[bb - 1] else df['low'].iloc[bb + 1]
        r_yLoc.iloc[i] = df['high'].iloc[bb - 1] if df['high'].iloc[bb + 1] > df['high'].iloc[bb - 1] else df['high'].iloc[bb + 1]

        # Support box (from pl pivot)
        pl_idx = df.index.get_loc(df.index[i]) if i in df.index else i
        # Find last valid pl before i
        pl_vals = pl.iloc[:i].dropna()
        if len(pl_vals) > 0:
            last_pl_idx = pl_vals.index[-1]
            last_pl_pos = df.index.get_loc(last_pl_idx)
            sBot.iloc[i] = pl.iloc[last_pl_pos]
            sTop.iloc[i] = pl.iloc[last_pl_pos]

        # Resistance box (from ph pivot)
        ph_vals = ph.iloc[:i].dropna()
        if len(ph_vals) > 0:
            last_ph_idx = ph_vals.index[-1]
            last_ph_pos = df.index.get_loc(last_ph_idx)
            rBot.iloc[i] = ph.iloc[last_ph_pos]
            rTop.iloc[i] = ph.iloc[last_ph_pos]

    # Calculate crossover/crossunder
    cu = pd.Series(False, index=df.index)  # crossunder (breakout down through support)
    co = pd.Series(False, index=df.index)  # crossover (breakout up through resistance)

    for i in range(1, len(df)):
        if pd.notna(sBot.iloc[i]) and pd.notna(sBot.iloc[i-1]):
            if input_repType == 'On':
                cu.iloc[i] = df['close'].iloc[i] < sBot.iloc[i] and df['close'].iloc[i-1] >= sBot.iloc[i-1]
            elif input_repType == 'Off: High & Low':
                cu.iloc[i] = df['low'].iloc[i] < sBot.iloc[i] and df['low'].iloc[i-1] >= sBot.iloc[i-1]
            else:  # 'Off: Candle Confirmation'
                cu.iloc[i] = df['close'].iloc[i] < sBot.iloc[i] and df['close'].iloc[i-1] >= sBot.iloc[i-1] and i == len(df) - 1

        if pd.notna(rTop.iloc[i]) and pd.notna(rTop.iloc[i-1]):
            if input_repType == 'On':
                co.iloc[i] = df['close'].iloc[i] > rTop.iloc[i] and df['close'].iloc[i-1] <= rTop.iloc[i-1]
            elif input_repType == 'Off: High & Low':
                co.iloc[i] = df['high'].iloc[i] > rTop.iloc[i] and df['high'].iloc[i-1] <= rTop.iloc[i-1]
            else:  # 'Off: Candle Confirmation'
                co.iloc[i] = df['close'].iloc[i] > rTop.iloc[i] and df['close'].iloc[i-1] <= rTop.iloc[i-1] and i == len(df) - 1

    # Detect breakout events
    sBreak = pd.Series(False, index=df.index)  # support breakout occurred
    rBreak = pd.Series(False, index=df.index)  # resistance breakout occurred

    for i in range(1, len(df)):
        if cu.iloc[i] and not sBreak.iloc[i-1]:
            sBreak.iloc[i] = True
        elif pd.notna(pl.iloc[i]) and pl.iloc[i] != pl.iloc[i-1]:
            if not sBreak.iloc[i-1]:
                sBreak.iloc[i] = False
            else:
                sBreak.iloc[i] = True
        else:
            sBreak.iloc[i] = sBreak.iloc[i-1]

        if co.iloc[i] and not rBreak.iloc[i-1]:
            rBreak.iloc[i] = True
        elif pd.notna(ph.iloc[i]) and ph.iloc[i] != ph.iloc[i-1]:
            if not rBreak.iloc[i-1]:
                rBreak.iloc[i] = False
            else:
                rBreak.iloc[i] = True
        else:
            rBreak.iloc[i] = rBreak.iloc[i-1]

    # Calculate barssince for breakout
    barssince_sBreak = pd.Series(np.nan, index=df.index)
    barssince_rBreak = pd.Series(np.nan, index=df.index)

    for i in range(len(df)):
        if sBreak.iloc[i]:
            barssince_sBreak.iloc[i] = 0
        else:
            # Find bars since last True
            for j in range(i-1, -1, -1):
                if sBreak.iloc[j]:
                    barssince_sBreak.iloc[i] = i - j
                    break

    for i in range(len(df)):
        if rBreak.iloc[i]:
            barssince_rBreak.iloc[i] = 0
        else:
            for j in range(i-1, -1, -1):
                if rBreak.iloc[j]:
                    barssince_rBreak.iloc[i] = i - j
                    break

    # Retest conditions
    sRetValid = pd.Series(False, index=df.index)
    rRetValid = pd.Series(False, index=df.index)

    for i in range(bb + 1, len(df)):
        if barssince_sBreak.iloc[i] > input_retSince and pd.notna(sTop.iloc[i]) and pd.notna(sBot.iloc[i]):
            # s1: high >= sTop and close <= sBot
            cond_s1 = df['high'].iloc[i] >= sTop.iloc[i] and df['close'].iloc[i] <= sBot.iloc[i]
            # s2: high >= sTop and close >= sBot and close <= sTop
            cond_s2 = df['high'].iloc[i] >= sTop.iloc[i] and df['close'].iloc[i] >= sBot.iloc[i] and df['close'].iloc[i] <= sTop.iloc[i]
            # s3: high >= sBot and high <= sTop
            cond_s3 = df['high'].iloc[i] >= sBot.iloc[i] and df['high'].iloc[i] <= sTop.iloc[i]
            # s4: high >= sBot and high <= sTop and close < sBot
            cond_s4 = df['high'].iloc[i] >= sBot.iloc[i] and df['high'].iloc[i] <= sTop.iloc[i] and df['close'].iloc[i] < sBot.iloc[i]

            sActive = cond_s1 or cond_s2 or cond_s3 or cond_s4
            retSince = barssince_sBreak.iloc[i]

            if sActive and retSince > 0 and retSince <= input_retValid:
                sRetValid.iloc[i] = True

        if barssince_rBreak.iloc[i] > input_retSince and pd.notna(rTop.iloc[i]) and pd.notna(rBot.iloc[i]):
            # r1: low <= rBot and close >= rTop
            cond_r1 = df['low'].iloc[i] <= rBot.iloc[i] and df['close'].iloc[i] >= rTop.iloc[i]
            # r2: low <= rBot and close <= rTop and close >= rBot
            cond_r2 = df['low'].iloc[i] <= rBot.iloc[i] and df['close'].iloc[i] <= rTop.iloc[i] and df['close'].iloc[i] >= rBot.iloc[i]
            # r3: low <= rTop and low >= rBot
            cond_r3 = df['low'].iloc[i] <= rTop.iloc[i] and df['low'].iloc[i] >= rBot.iloc[i]
            # r4: low <= rTop and low >= rBot and close > rTop
            cond_r4 = df['low'].iloc[i] <= rTop.iloc[i] and df['low'].iloc[i] >= rBot.iloc[i] and df['close'].iloc[i] > rTop.iloc[i]

            rActive = cond_r1 or cond_r2 or cond_r3 or cond_r4
            retSince_r = barssince_rBreak.iloc[i]

            if rActive and retSince_r > 0 and retSince_r <= input_retValid:
                rRetValid.iloc[i] = True

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i == 0:
            continue

        # Skip if indicators are NaN
        if pd.isna(sTop.iloc[i]) or pd.isna(sBot.iloc[i]) or pd.isna(rTop.iloc[i]) or pd.isna(rBot.iloc[i]):
            continue

        if sRetValid.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

        if rRetValid.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries