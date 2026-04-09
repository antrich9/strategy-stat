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
    n = len(df)
    entries = []
    trade_num = 0

    # Volume filter
    vol_sma9 = df['volume'].rolling(9).mean()
    volfilt_raw = df['volume'].shift(1) > vol_sma9 * 1.5

    # ATR filter - Wilder ATR
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr20_raw = tr.ewm(alpha=1/20, adjust=False).mean()
    atr = atr20_raw / 1.5

    # Trend filter
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    # Bullish FVG condition
    bfvg = (low > high.shift(2)) & volfilt_raw & ((low - high.shift(2) > atr) | (low.shift(2) - high > atr)) & locfiltb

    # Bearish FVG condition
    sfvg = (high < low.shift(2)) & volfilt_raw & ((low - high.shift(2) > atr) | (low.shift(2) - high > atr)) & locfilts

    # Identify order blocks
    def is_up(idx):
        return close.iloc[idx] > df['open'].iloc[idx]

    def is_down(idx):
        return close.iloc[idx] < df['open'].iloc[idx]

    # OB detection
    ob_up = pd.Series(False, index=df.index)
    ob_down = pd.Series(False, index=df.index)

    for i in range(2, n):
        if is_down(i - 1) and is_up(i) and close.iloc[i] > high.iloc[i - 1]:
            ob_up.iloc[i] = True
        if is_up(i - 1) and is_down(i) and close.iloc[i] < low.iloc[i - 1]:
            ob_down.iloc[i] = True

    # FVG detection
    fvg_up = low > high.shift(2)
    fvg_down = high < low.shift(2)

    # Stacked OB + FVG conditions
    stacked_bullish = ob_up & fvg_up
    stacked_bearish = ob_down & fvg_down

    # Entry conditions
    long_cond = stacked_bullish
    short_cond = stacked_bearish

    # Previous bar values for crossover/crossunder
    long_cond_prev = long_cond.shift(1).fillna(False)
    short_cond_prev = short_cond.shift(1).fillna(False)

    for i in range(2, n):
        if pd.isna(bfvg.iloc[i]) or pd.isna(sfvg.iloc[i]):
            continue
        if long_cond.iloc[i] and not long_cond_prev.iloc[i]:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
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
        if short_cond.iloc[i] and not short_cond_prev.iloc[i]:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
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

    return entries