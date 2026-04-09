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
    # Default parameters from Pine Script
    zlFast = 12
    zlSlow = 26
    zlSignal = 9
    ttmsLen = 20
    ttmsBBMult = 2.0
    requireBothCross = True

    close = df['close']
    high = df['high']
    low = df['low']

    # Zero Lag MACD calculation
    zlema_fast = close.ewm(span=zlFast, adjust=False).mean()
    zlema_slow = close.ewm(span=zlSlow, adjust=False).mean()
    zlMACD = zlema_fast - zlema_slow
    zlSignalLine = zlMACD.ewm(span=zlSignal, adjust=False).mean()

    # TTMS calculation
    smma = close.ewm(alpha=1.0/ttmsLen, adjust=False).mean()
    highest_high = high.rolling(ttmsLen).max()
    lowest_low = low.rolling(ttmsLen).min()
    midline = (highest_high + lowest_low) / 2
    TTMS_raw = (smma - midline) / (highest_high - lowest_low + 1e-10)
    TTMS_smooth = TTMS_raw.ewm(span=3, adjust=False).mean()

    # Bullish/Bearish conditions
    zl_bullish = zlMACD > zlSignalLine
    zl_bearish = zlMACD < zlSignalLine
    ttms_bullish = TTMS_smooth > 0
    ttms_bearish = TTMS_smooth < 0

    # Crossover detection
    zl_crossup = (zlMACD > zlSignalLine) & (zlMACD.shift(1) <= zlSignalLine.shift(1))
    zl_crossdown = (zlMACD < zlSignalLine) & (zlMACD.shift(1) >= zlSignalLine.shift(1))
    ttms_crossup = (TTMS_smooth > 0) & (TTMS_smooth.shift(1) <= 0)
    ttms_crossdown = (TTMS_smooth < 0) & (TTMS_smooth.shift(1) >= 0)

    # Entry conditions based on requireBothCross
    long_condition = pd.Series(False, index=df.index)
    short_condition = pd.Series(False, index=df.index)

    if requireBothCross:
        long_condition = zl_crossup & ttms_crossup
        short_condition = zl_crossdown & ttms_crossdown
    else:
        long_condition = zl_bullish & ttms_bullish
        short_condition = zl_bearish & ttms_bearish

    entries = []
    trade_num = 1

    for i in range(2, len(df)):
        if pd.isna(zlMACD.iloc[i]) or pd.isna(TTMS_smooth.iloc[i]):
            continue

        direction = None
        if long_condition.iloc[i]:
            direction = 'long'
        elif short_condition.iloc[i]:
            direction = 'short'

        if direction:
            entry_price = close.iloc[i]
            entry_row = df.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': int(entry_row['time']),
                'entry_time': datetime.fromtimestamp(entry_row['time'], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

    return entries