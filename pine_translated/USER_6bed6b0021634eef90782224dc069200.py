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
    # --- compute needed series -------------------------------------------------
    high = df['high']
    low  = df['low']
    close = df['close']

    # shifted series
    high_1 = high.shift(1)
    high_2 = high.shift(2)
    low_1  = low.shift(1)
    low_2  = low.shift(2)
    close_1 = close.shift(1)

    # gap detection
    bullG = low > high_1          # bullish gap (low breaks previous high)
    bearG = high < low_1          # bearish gap (high breaks previous low)

    # True Range and Wilder ATR (length 144)
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - close_1),
                               np.abs(low - close_1)))
    atr = tr.ewm(alpha=1/144, adjust=False).mean().fillna(0)

    # width filter factor (default 0.5)
    fvg_th = 0.5
    atr_th = atr * fvg_th

    # bull entry condition
    bull = ((low - high_2) > atr_th) & \
           (low > high_2) & \
           (close_1 > high_2) & \
           ~(bullG | bullG.shift(1))

    # bear entry condition
    bear = ((low_2 - high) > atr_th) & \
           (high < low_2) & \
           (close_1 < low_2) & \
           ~(bearG | bearG.shift(1))

    # ensure no NaN booleans slip through
    bull = bull.fillna(False)
    bear = bear.fillna(False)

    # --- generate entries -----------------------------------------------------
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if bull.iloc[i]:
            direction = 'long'
        elif bear.iloc[i]:
            direction = 'short'
        else:
            continue

        entry_ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        entry_price = float(df['close'].iloc[i])

        entries.append({
            'trade_num': trade_num,
            'direction': direction,
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