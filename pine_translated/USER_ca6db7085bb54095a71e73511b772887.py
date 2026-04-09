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
    # Parameters from Pine Script inputs
    L = 30
    vzLen = 60
    vzThr = -0.5
    rLen = 60
    fadeNoLow = 0.01
    longThr = 1.0
    shortThr = -1.0

    # --- Calculate indicators ---
    # Volume trend
    vol_trend = df['volume'].rolling(L).mean()

    # Volume shock (vs)
    vs = pd.Series(np.where(
        (df['volume'] > 0) & (vol_trend > 0),
        np.log(df['volume']) - np.log(vol_trend),
        np.nan
    ), index=df.index)

    # Z-score using Wilder EWM method
    def wilder_zscore(series, length):
        ewm_mean = series.ewm(span=length, adjust=False).mean()
        ewm_std = series.ewm(span=length, adjust=False).mean()
        abs_dev = (series - ewm_mean).abs()
        ewm_abs_dev = abs_dev.ewm(span=length, adjust=False).mean()
        return (series - ewm_mean) / ewm_abs_dev

    vsZ = wilder_zscore(vs, vzLen)

    # Return and z-score
    r1 = np.log(df['close'] / df['close'].shift(1))
    retZ = wilder_zscore(r1, rLen)

    # Oscillator
    low_vs = vsZ <= vzThr
    weight = np.where(low_vs, 1.0, fadeNoLow)
    osc = retZ * weight

    # FVG detection
    bullish_fvg = df['low'] > df['high'].shift(2)
    bearish_fvg = df['high'] < df['low'].shift(2)

    # Track FVG state
    last_fvg = 0
    bullsharp = False
    bearsharp = False

    entries = []
    trade_num = 1

    # Iterate bars starting from index 2 for FVG
    for i in range(2, len(df)):
        # Update FVG state
        bfvg = bullish_fvg.iloc[i]
        sfvg = bearish_fvg.iloc[i]

        if bfvg and last_fvg == -1:
            last_fvg = 1
            bullsharp = True
            bearsharp = False
        elif sfvg and last_fvg == 1:
            last_fvg = -1
            bearsharp = True
            bullsharp = False
        elif bfvg:
            last_fvg = 1
        elif sfvg:
            last_fvg = -1

        # Check entries (skip if NaN)
        if pd.isna(osc.iloc[i]):
            continue

        entry_long = bullsharp and (osc.iloc[i] >= longThr)
        entry_short = bearsharp and (osc.iloc[i] <= shortThr)

        if entry_long:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif entry_short:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1

    return entries