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
    # default filter settings (same as Pine script inputs)
    inp1 = False  # Volume Filter
    inp2 = False  # ATR Filter
    inp3 = False  # Trend Filter

    # Volume filter
    vol_sma9 = df['volume'].rolling(9).mean()
    vol_filt = df['volume'].shift(1) > vol_sma9 * 1.5
    if not inp1:
        vol_filt = pd.Series(True, index=df.index)

    # ATR (Wilder)
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(np.abs(df['high'] - df['close'].shift(1)),
                               np.abs(df['low'] - df['close'].shift(1))))
    atr_raw = tr.ewm(alpha=1/20, adjust=False).mean()
    atr = atr_raw / 1.5

    # ATR filter
    atrfilt = (df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr)
    if not inp2:
        atrfilt = pd.Series(True, index=df.index)

    # Trend filter
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2.copy()
    locfilts = ~loc2
    if not inp3:
        locfiltb = pd.Series(True, index=df.index)
        locfilts = pd.Series(True, index=df.index)

    # Bullish FVG (long entry)
    bfvg = (df['low'] > df['high'].shift(2)) & vol_filt & atrfilt & locfiltb
    # Bearish FVG (short entry)
    sfvg = (df['high'] < df['low'].shift(2)) & vol_filt & atrfilt & locfilts

    entries = []
    trade_num = 1

    # need at least 2 bars for shift(2)
    for i in range(2, len(df)):
        # skip if any indicator is NaN (should be False anyway)
        if bfvg.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif sfvg.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries