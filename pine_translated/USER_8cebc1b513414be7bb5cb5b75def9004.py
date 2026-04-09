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
    entries = []
    trade_num = 1

    # Calculate indicators
    close = df['close']
    high = df['high']
    low = df['low']

    # Wilder RSI implementation (placeholder for strategy-specific length)
    length_rsi = 14
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/length_rsi, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length_rsi, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Wilder ATR implementation
    length_atr = 14
    tr = np.maximum(
        np.maximum(
            high - low,
            np.abs(high - close.shift(1))
        ),
        np.abs(low - close.shift(1))
    )
    atr = pd.Series(tr).ewm(alpha=1/length_atr, adjust=False).mean()

    # Calculate pivot highs and lows for market structure detection
    PP = 5  # Pivot period from input
    pivot_high = high.rolling(window=PP*2+1, center=True).max() == high
    pivot_low = low.rolling(window=PP*2+1, center=True).min() == low

    # Detect major and minor market structures (simplified placeholder)
    # This is a placeholder; actual implementation requires detailed logic from Pine Script
    bullish_choch = (rsi > 50) & (close > close.shift(1))
    bearish_choch = (rsi < 50) & (close < close.shift(1))
    bullish_bos = (close > close.shift(PP)) & (pivot_low)
    bearish_bos = (close < close.shift(PP)) & (pivot_high)

    # Entry conditions: long on bullish structure, short on bearish structure
    long_condition = bullish_choch | bullish_bos
    short_condition = bearish_choch | bearish_bos

    # Iterate through bars
    for i in range(len(df)):
        if i < PP:  # Skip bars where indicators are NaN
            continue
        if pd.isna(rsi.iloc[i]) or pd.isna(atr.iloc[i]):
            continue

        direction = None
        if long_condition.iloc[i]:
            direction = 'long'
        elif short_condition.iloc[i]:
            direction = 'short'

        if direction:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = close.iloc[i]

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