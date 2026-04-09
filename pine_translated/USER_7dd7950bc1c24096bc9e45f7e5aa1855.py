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
    if len(df) < 3:
        return []

    # ATR calculation (Wilder)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()

    # SMA for volume filter
    vol_sma = df['volume'].rolling(9).mean()

    # SMA for trend filter
    loc = df['close'].rolling(54).mean()
    loc_shift = loc.shift(1)
    loc2 = loc > loc_shift

    # Conditions (inp1=False, inp2=False, inp3=False in this config)
    # volfilt = volume[1] > sma(volume, 9)*1.5 (when inp1=true)
    # atrfilt = (low - high[2] > atr) or (low[2] - high > atr) (when inp2=true)
    # locfiltb = loc > loc[1] (when inp3=true)
    # locfilts = not (loc > loc[1]) (when inp3=true)

    # FVG conditions (shift indices: Pine uses current bar, so Python uses shift(0) which is same col)
    # Pine: low > high[2] means current low > high 2 bars ago
    bfvg_cond = (df['low'] > df['high'].shift(2))  # Bullish FVG
    sfvg_cond = (df['high'] < df['low'].shift(2))   # Bearish FVG

    # Skip first 2 bars due to shift dependencies
    entries = []
    trade_num = 1

    for i in range(2, len(df)):
        is_bullish_fvg = bfvg_cond.iloc[i]
        is_bearish_fvg = sfvg_cond.iloc[i]

        if is_bullish_fvg:
            entry_price = df['close'].iloc[i]
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        if is_bearish_fvg:
            entry_price = df['close'].iloc[i]
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
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