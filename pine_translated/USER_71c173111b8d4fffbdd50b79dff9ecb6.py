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
    # Pattern parameters (default values from the Pine Script inputs)
    bearish_candle_strength = 0.7
    bullish_strike_strength = 1.5

    # Body size and range for each bar
    body_size = (df['close'] - df['open']).abs()
    candle_range = df['high'] - df['low']

    # Black (bearish) candle condition
    black_candle = (df['close'] < df['open']) & (body_size > candle_range * bearish_candle_strength)

    # Shifted black candles for the three consecutive bars
    black1 = black_candle.shift(3)
    black2 = black_candle.shift(2)
    black3 = black_candle.shift(1)

    # Bullish strike candle condition for the current bar
    bullish_strike = (
        (df['close'] > df['open']) &
        (body_size > candle_range * bullish_strike_strength) &
        (df['close'] > df['close'].shift(3))
    )

    # Progressively lower lows
    progressively_lower_lows = (
        (df['low'].shift(1) < df['low'].shift(2)) &
        (df['low'].shift(2) < df['low'].shift(3))
    )

    # Volume confirmation
    volume_confirmation = (
        (df['volume'].shift(3) > df['volume'].shift(2)) &
        (df['volume'].shift(2) > df['volume'].shift(1)) &
        (df['volume'] > df['volume'].shift(1))
    )

    # Downtrend confirmation
    in_downtrend = (
        (df['close'].shift(3) < df['close'].shift(4)) &
        (df['close'].shift(2) < df['close'].shift(3)) &
        (df['close'].shift(1) < df['close'].shift(2))
    )

    # Full pattern condition
    pattern = (
        black1 & black2 & black3 &
        bullish_strike &
        progressively_lower_lows &
        in_downtrend &
        volume_confirmation
    )
    # Ensure NaN values are treated as False
    pattern = pattern.fillna(False)

    entries = []
    trade_num = 1
    for i in range(len(df)):
        if pattern.iloc[i]:
            ts = int(df.iloc[i]['time'])
            entry_price = float(df.iloc[i]['close'])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price,
            })
            trade_num += 1

    return entries