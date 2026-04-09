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
    # Ensure required columns present
    required = {'time', 'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns):
        raise ValueError(f"df must contain columns {required}")
    # Make copies to avoid modifying original
    high = df['high'].copy()
    low = df['low'].copy()
    close = df['close'].copy()
    volume = df['volume'].copy()
    time = df['time'].copy()

    # Compute hour in UTC (assuming timestamps are in seconds)
    ts_dt = pd.to_datetime(time, unit='s', utc=True)
    hour = ts_dt.dt.hour

    # Valid trade time windows
    is_valid_time = ((hour >= 2) & (hour < 5)) | ((hour >= 10) & (hour < 12))

    # Optional filters (default disabled)
    # Volume filter: volume[1] > sma(volume,9)*1.5
    vol_sma = volume.rolling(9).mean()
    vol_cond = volume.shift(1) > vol_sma * 1.5
    volfilt = vol_cond.fillna(True)  # default disabled -> True

    # ATR filter (20 period Wilder ATR)
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atr = atr / 1.5
    atrfilt = ((low - high.shift(2) > atr) | (low.shift(2) - high > atr)).fillna(True)  # default disabled

    # Trend filter: SMA 54 direction
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2.fillna(True)  # default disabled -> True
    locfilts = (~loc2).fillna(True)  # default disabled -> True

    # Bullish breakaway FVG condition
    bfvg = (low > high.shift(2)) & volfilt & atrfilt & locfiltb
    # Bearish breakaway FVG condition
    sfvg = (high < low.shift(2)) & volfilt & atrfilt & locfilts

    # Fill NaNs with False
    bfvg = bfvg.fillna(False)
    sfvg = sfvg.fillna(False)
    is_valid_time = is_valid_time.fillna(False)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if bfvg.iloc[i] and is_valid_time.iloc[i]:
            ts = int(time.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        elif sfvg.iloc[i] and is_valid_time.iloc[i]:
            ts = int(time.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

    return entries