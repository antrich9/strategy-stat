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

    # Default parameters from strategy
    left = 20
    right = 15
    nPiv = 4
    atrLen = 30

    # Detection defaults (all false by default)
    detectBO = False
    detectBD = False
    breakUp = False
    breakDn = False
    falseBull = False
    falseBear = False
    supPush = False
    resPush = False
    curl = False

    entries = []
    trade_num = 1

    # Helper for Wilder RSI
    def WilderRSI(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # Helper for Wilder ATR
    def WilderATR(high, low, close, length):
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        return atr

    # Calculate ATR
    atr = WilderATR(df['high'], df['low'], df['close'], atrLen)

    # Detect pivot highs and lows
    pivot_high = pd.Series(False, index=df.index)
    pivot_low = pd.Series(False, index=df.index)

    for i in range(left + right, len(df)):
        if df['high'].iloc[i] == df['high'].iloc[i-left:i+right+1].max():
            pivot_high.iloc[i] = True
        if df['low'].iloc[i] == df['low'].iloc[i-left:i+right+1].min():
            pivot_low.iloc[i] = True

    # Track recent pivot highs and lows
    hi_tracks = []
    lo_tracks = []
    max_high = pd.Series(np.nan, index=df.index)
    min_low = pd.Series(np.nan, index=df.index)

    for i in range(len(df)):
        if pivot_high.iloc[i]:
            hi_tracks.append(df['high'].iloc[i])
            if len(hi_tracks) > nPiv:
                hi_tracks.pop(0)
        if pivot_low.iloc[i]:
            lo_tracks.append(df['low'].iloc[i])
            if len(lo_tracks) > nPiv:
                lo_tracks.pop(0)

        if hi_tracks:
            max_high.iloc[i] = max(hi_tracks)
        if lo_tracks:
            min_low.iloc[i] = min(lo_tracks)

    # Build entry conditions
    valid_lookback = max(left, right, atrLen) + nPiv

    for i in range(valid_lookback, len(df)):
        if pd.isna(max_high.iloc[i]) or pd.isna(min_low.iloc[i]) or pd.isna(atr.iloc[i]):
            continue

        breakout_long = df['close'].iloc[i] > max_high.iloc[i] and df['close'].iloc[i-1] <= max_high.iloc[i-1] if not pd.isna(max_high.iloc[i-1]) else False
        breakdown_short = df['close'].iloc[i] < min_low.iloc[i] and df['close'].iloc[i-1] >= min_low.iloc[i-1] if not pd.isna(min_low.iloc[i-1]) else False

        long_entry = (detectBO and breakout_long) or (breakUp and breakout_long)
        short_entry = (detectBD and breakdown_short) or (breakDn and breakdown_short)

        if long_entry:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

        if short_entry:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries