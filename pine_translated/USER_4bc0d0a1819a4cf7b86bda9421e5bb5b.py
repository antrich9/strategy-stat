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
        raise ValueError(f"DataFrame must contain columns: {required}")

    # Input flags (hardcoded as per script)
    inp1 = False  # Volume Filter
    inp2 = False  # ATR Filter
    inp3 = False  # Trend Filter

    # Volume filter
    if inp1:
        vol_filter = (df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5)
    else:
        vol_filter = pd.Series(True, index=df.index)

    # Wilder ATR computation (period 20)
    def wilder_atr(high, low, close, period=20):
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # Fill NaN for first bar with tr1 (high - low)
        tr.fillna(tr1, inplace=True)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr

    # ATR filter
    if inp2:
        atr = wilder_atr(df['high'], df['low'], df['close'], period=20) / 1.5
        atrfilt = ((df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr))
    else:
        atrfilt = pd.Series(True, index=df.index)

    # Trend filter
    if inp3:
        loc = df['close'].rolling(54).mean()
        loc2 = loc > loc.shift(1)
        locfiltb = loc2
        locfilts = ~loc2
    else:
        locfiltb = pd.Series(True, index=df.index)
        locfilts = pd.Series(True, index=df.index)

    # Bullish Fair Value Gap (FVG) – low > high 2 bars ago
    bfvg = (df['low'] > df['high'].shift(2)) & vol_filter & atrfilt & locfiltb

    # Bearish Fair Value Gap – high < low 2 bars ago
    sfvg = (df['high'] < df['low'].shift(2)) & vol_filter & atrfilt & locfilts

    # Generate entry list
    entries = []
    trade_num = 1

    # Iterate over each row (skip first two bars where shift(2) is NaN)
    for i in df.index:
        if i < 2:
            continue

        # Long entry
        if bfvg.loc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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

        # Short entry
        if sfvg.loc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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