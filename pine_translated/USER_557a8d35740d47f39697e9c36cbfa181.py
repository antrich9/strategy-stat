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
    # Lagged high/low for FVG detection
    high_lag2 = df['high'].shift(2)
    low_lag2 = df['low'].shift(2)

    # Volume filter (default input is false – filter always passes)
    vol = df['volume']
    vol_sma = vol.rolling(9).mean()
    volfilt = vol.shift(1) > vol_sma * 1.5

    # ATR filter (default input is false – filter always passes)
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(np.abs(df['high'] - df['close'].shift(1)),
                               np.abs(df['low'] - df['close'].shift(1))))
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atrfilt = (df['low'] - high_lag2 > atr) | (low_lag2 - df['high'] > atr)

    # Trend filter (default input is false – filter always passes)
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    # Entry conditions
    long_cond = (df['low'] > high_lag2) & volfilt & atrfilt & locfiltb
    short_cond = (df['high'] < low_lag2) & volfilt & atrfilt & locfilts

    entries = []
    trade_num = 1

    for i in range(len(df)):
        # Skip bars where required indicators are NaN (first two bars)
        if i < 2:
            continue

        # Long entry
        if not pd.isna(long_cond.iloc[i]) and long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

        # Short entry
        if not pd.isna(short_cond.iloc[i]) and short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries