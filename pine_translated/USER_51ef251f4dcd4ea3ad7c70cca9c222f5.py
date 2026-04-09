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
    # Alligator parameters
    jaw_len = 13
    teeth_len = 8
    lips_len = 5

    # ADX parameters
    adx_len = 14
    adx_threshold = 30.0

    # Alligator moving averages
    jaw = df['close'].rolling(window=jaw_len).mean()
    teeth = df['close'].rolling(window=teeth_len).mean()
    lips = df['close'].rolling(window=lips_len).mean()

    # ADX components
    up_move = df['high'].diff(1)                      # high - high[1]
    down_move = df['low'].shift(1) - df['low']        # low[1] - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index
    )

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)

    smoothed_tr = tr.rolling(window=adx_len).mean()
    smoothed_plus_dm = plus_dm.rolling(window=adx_len).mean()
    smoothed_minus_dm = minus_dm.rolling(window=adx_len).mean()

    # DI calculations (protect against division by zero)
    plus_di = pd.Series(
        np.where(smoothed_tr != 0, (smoothed_plus_dm / smoothed_tr) * 100, np.nan),
        index=df.index
    )
    minus_di = pd.Series(
        np.where(smoothed_tr != 0, (smoothed_minus_dm / smoothed_tr) * 100, np.nan),
        index=df.index
    )

    # DX and ADX
    dx = pd.Series(
        np.where(
            (plus_di + minus_di) != 0,
            np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100,
            np.nan
        ),
        index=df.index
    )
    adx = dx.rolling(window=adx_len).mean()

    # Entry conditions (ensure NaN becomes False)
    long_cond = (
        lips.gt(teeth) & teeth.gt(jaw) &
        df['close'].gt(lips) &
        adx.gt(adx_threshold)
    ).fillna(False)

    short_cond = (
        lips.lt(teeth) & teeth.lt(jaw) &
        df['close'].lt(lips) &
        adx.gt(adx_threshold)
    ).fillna(False)

    # Build entry list
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if long_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries