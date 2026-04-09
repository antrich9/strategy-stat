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
    # Ensure chronological order
    df = df.sort_values('time').reset_index(drop=True)

    high = df['high']
    low = df['low']
    close = df['close']

    # ----- True Range -----
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr.iloc[0] = high.iloc[0] - low.iloc[0]  # first bar has no prior close

    # ----- Wilder ATR (period = 10) -----
    atr_period = 10
    atr = pd.Series(index=df.index, dtype=float)
    atr.iloc[0] = tr.iloc[0]
    for i in range(1, len(df)):
        atr.iloc[i] = (atr.iloc[i - 1] * (atr_period - 1) + tr.iloc[i]) / atr_period

    # ----- SuperTrend source (hl2) -----
    src = (high + low) / 2.0

    # ----- SuperTrend bands -----
    multiplier = 3.0
    up = src - multiplier * atr
    dn = src + multiplier * atr
    up1 = up.shift(1).fillna(up)
    dn1 = dn.shift(1).fillna(dn)

    # ----- Trend calculation -----
    trend = pd.Series(index=df.index, dtype=int)
    trend.iloc[0] = 1
    prev_trend = 1
    for i in range(1, len(df)):
        if prev_trend == -1 and close.iloc[i] > dn1.iloc[i]:
            cur_trend = 1
        elif prev_trend == 1 and close.iloc[i] < up1.iloc[i]:
            cur_trend = -1
        else:
            cur_trend = prev_trend
        trend.iloc[i] = cur_trend
        prev_trend = cur_trend

    # ----- Entry signals -----
    buy_signal = (trend == 1) & (trend.shift(1) == -1)
    sell_signal = (trend == -1) & (trend.shift(1) == 1)

    # ----- Build entry list -----
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if np.isnan(atr.iloc[i]):
            continue
        if buy_signal.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif sell_signal.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries