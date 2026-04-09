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
    adx_length = 14
    adx_threshold = 25

    results = []
    trade_num = 0

    high = df['high']
    low = df['low']
    close = df['close']

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothed ATR
    alpha = 1.0 / adx_length
    atr = tr.ewm(alpha=alpha, adjust=False).mean()

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(np.where(up_move > down_move, up_move.values, 0.0), index=df.index)
    minus_dm = pd.Series(np.where(down_move > up_move, down_move.values, 0.0), index=df.index)

    # Smooth directional indicators
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    # Directional Indicators
    di_plus_smooth = 100 * plus_dm_smooth / atr
    di_minus_smooth = 100 * minus_dm_smooth / atr

    # Directional Index
    di_sum = di_plus_smooth + di_minus_smooth
    dx = 100 * (di_plus_smooth - di_minus_smooth).abs() / di_sum.replace(0, np.nan)

    # ADX
    adx_series = dx.ewm(alpha=alpha, adjust=False).mean()

    # EMAs (simulating request.security for 5 min data)
    ema_fast = close.ewm(span=10, adjust=False).mean()
    ema_slow = close.ewm(span=50, adjust=False).mean()

    # London time windows (timestamps in UTC)
    morning_start_hour = 8
    morning_end_hour = 9
    afternoon_start_hour = 14
    afternoon_end_hour = 16

    for i in range(2, len(df)):
        if pd.isna(ema_fast.iloc[i]) or pd.isna(atr.iloc[i]):
            continue

        ts = df['time'].iloc[i]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour

        is_within_time_window = (
            (hour >= morning_start_hour and hour < morning_end_hour + 55) or
            (hour >= afternoon_start_hour and hour < afternoon_end_hour + 55)
        )

        if is_within_time_window:
            adx_val = adx_series.iloc[i]
            di_plus_val = di_plus_smooth.iloc[i]
            di_minus_val = di_minus_smooth.iloc[i]
            ema_fast_val = ema_fast.iloc[i]
            ema_slow_val = ema_slow.iloc[i]

            # Bullish FVG detection
            bullish_fvg = low.iloc[i-2] > high.iloc[i]
            # Bearish FVG detection
            bearish_fvg = high.iloc[i-2] < low.iloc[i]

            if adx_val > adx_threshold:
                # Long entry conditions
                if di_plus_val > di_minus_val and ema_fast_val > ema_slow_val and bullish_fvg:
                    trade_num += 1
                    entry_price = close.iloc[i]
                    results.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': int(ts),
                        'entry_time': dt.isoformat(),
                        'entry_price_guess': entry_price,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': entry_price,
                        'raw_price_b': entry_price
                    })

                # Short entry conditions
                if di_minus_val > di_plus_val and ema_fast_val < ema_slow_val and bearish_fvg:
                    trade_num += 1
                    entry_price = close.iloc[i]
                    results.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': int(ts),
                        'entry_time': dt.isoformat(),
                        'entry_price_guess': entry_price,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': entry_price,
                        'raw_price_b': entry_price
                    })

    return results