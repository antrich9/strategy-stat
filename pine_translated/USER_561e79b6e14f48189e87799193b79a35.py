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
    # Keep a copy of the close series and the time column
    close = df['close'].copy()
    time_col = df['time'].copy()

    # ==== Indicator parameters ====
    shortLength = 20
    longLength  = 25
    trendLength = 200

    # Simple moving averages on the lower timeframe
    maShort = close.rolling(window=shortLength).mean()
    maLong  = close.rolling(window=longLength).mean()
    maTrend = close.rolling(window=trendLength).mean()

    # ==== Higher‑timeframe (H4 = 240 minutes) SMA ====
    # Convert integer timestamps to datetime for resampling
    times = pd.to_datetime(time_col, unit='s', utc=True)

    # Build a temporary DataFrame for resampling
    df_h4 = pd.DataFrame({'time': times, 'close': close}).set_index('time')

    # Resample to 240‑minute bars and take the last close of each bar
    df_h4_res = df_h4.resample('240T').last()

    # 200‑period SMA on the H4 data
    maTrendH4 = df_h4_res['close'].rolling(window=trendLength).mean()

    # Bring the H4 SMA back to the original (lower‑timeframe) index via forward‑fill
    maTrendH4 = maTrendH4.reindex(times, method='ffill')

    # For any remaining NaNs (e.g., before the first H4 bar) fall back to the lower‑timeframe SMA
    maTrendH4 = maTrendH4.fillna(maTrend)

    # ==== Crossover / Crossunder helpers ====
    def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a > b) & (a.shift(1) <= b.shift(1))

    def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a < b) & (a.shift(1) >= b.shift(1))

    # ==== Entry conditions ====
    long_cond  = crossover(maShort, maLong) & (close > maTrendH4)
    short_cond = crossunder(maShort, maLong) & (close < maTrendH4)

    # ==== Build entry list ====
    entries = []
    trade_num = 1

    for i in range(len(df)):
        # Skip bars where any required indicator is NaN (condition will be False/NaN)
        if long_cond.iloc[i]:
            entry_ts = int(time_col.iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])

            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        elif short_cond.iloc[i]:
            entry_ts = int(time_col.iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])

            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries