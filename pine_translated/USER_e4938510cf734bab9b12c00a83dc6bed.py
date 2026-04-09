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

    if len(df) < 146:
        return []

    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']

    # Wilder ATR (144 period)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(np.nan, index=tr.index, dtype=float)
    atr.iloc[143] = tr.iloc[:144].mean()
    for i in range(144, len(tr)):
        atr.iloc[i] = (atr.iloc[i-1] * 143 + tr.iloc[i]) / 144

    # SMA for close (54 periods)
    sma_close = close.rolling(54).mean()

    # Volume filter (volfilt)
    vol_sma = volume.rolling(9).mean()
    volfilt = volume.shift(1) > vol_sma * 1.5

    # ATR filter (atrfilt)
    atr1 = atr / 1.5
    atrfilt = (low - high.shift(2) > atr1) | (low.shift(2) - high > atr1)

    # Trend filter (locfiltb, locfilts)
    loc = sma_close
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    # Bullish FVG (bfvg)
    bfvg = low > high.shift(2) & volfilt & atrfilt & locfiltb

    # Bearish FVG (sfvg)
    sfvg = high < low.shift(2) & volfilt & atrfilt & locfilts

    # Consecutive FVG tracking
    consecutive_bfvg = pd.Series(0, index=df.index)
    consecutive_sfvg = pd.Series(0, index=df.index)

    for i in range(2, len(df)):
        if bfvg.iloc[i]:
            consecutive_bfvg.iloc[i] = consecutive_bfvg.iloc[i-1] + 1
        else:
            consecutive_bfvg.iloc[i] = 0

        if sfvg.iloc[i]:
            consecutive_sfvg.iloc[i] = consecutive_sfvg.iloc[i-1] + 1
        else:
            consecutive_sfvg.iloc[i] = 0

    # Bull and Bear conditions
    bullG = low > high.shift(1)
    bearG = high < low.shift(1)

    bull = (
        ((low - high.shift(2)) > atr) &
        (low > high.shift(2)) &
        (close.shift(1) > high.shift(2)) &
        ~(bullG | bullG.shift(1))
    )

    bear = (
        ((low.shift(2) - high) > atr) &
        (high < low.shift(2)) &
        (close.shift(1) < low.shift(2)) &
        ~(bearG | bearG.shift(1))
    )

    # Entry conditions
    bullish_entry = bull.shift(1) & (consecutive_bfvg == 2)
    bearish_entry = bear.shift(1) & (consecutive_sfvg == 2)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if bullish_entry.iloc[i] and consecutive_bfvg.iloc[i] == 2:
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif bearish_entry.iloc[i] and consecutive_sfvg.iloc[i] == 2:
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries