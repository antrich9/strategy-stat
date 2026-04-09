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
    # Ensure DataFrame is sorted by time
    df = df.sort_values('time').reset_index(drop=True)

    # Extract hour from timestamp for valid trade time check
    hour_series = pd.to_datetime(df['time'], unit='s', utc=True).dt.hour

    # Valid trade time: 02:00-04:59 or 10:00-11:59
    is_valid_time = ((hour_series >= 2) & (hour_series < 5)) | ((hour_series >= 10) & (hour_series < 12))

    # Wilder RSI implementation (length=14)
    def compute_rsi(series, length=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = compute_rsi(df['close'], length=14)

    # Bullish Fair Value Gap (FVG) detection
    # Conditions:
    # low > high[2]
    # close[1] > high[2]
    # open[2] < close[2]
    # open[1] < close[1]
    # open < close
    bull_fvg = (
        (df['low'] > df['high'].shift(2)) &
        (df['close'].shift(1) > df['high'].shift(2)) &
        (df['open'].shift(2) < df['close'].shift(2)) &
        (df['open'].shift(1) < df['close'].shift(1)) &
        (df['open'] < df['close'])
    )
    bull_fvg = bull_fvg.fillna(False).astype(bool)

    # Entry condition: bull_fvg AND valid time AND RSI > 50
    entry_cond = bull_fvg & is_valid_time & (rsi > 50)

    entries = []
    trade_num = 1
    for i in range(len(df)):
        if entry_cond.iloc[i]:
            ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            # Strategy uses a stop order at the low of the bar
            entry_price = df['low'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
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