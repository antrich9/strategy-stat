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
    if len(df) < 50:
        return []

    SwingPeriod = 50
    ATR_Coefficient = 1.0
    FVG_Length = 150
    MSS_Length = 150

    high = df['high']
    low = df['low']
    close = df['close']
    open_arr = df['open']

    tr1 = high - low
    tr2 = abs(high - close.shift(1).fillna(close))
    tr3 = abs(low - close.shift(1).fillna(close))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()

    pivot_high = pd.Series(index=df.index, dtype=float)
    pivot_low = pd.Series(index=df.index, dtype=float)

    for i in range(SwingPeriod, len(df) - SwingPeriod):
        ph = high.iloc[i - SwingPeriod:i + SwingPeriod + 1].max()
        pl = low.iloc[i - SwingPeriod:i + SwingPeriod + 1].min()
        pivot_high.iloc[i] = ph
        pivot_low.iloc[i] = pl

    swing_high = pd.Series(False, index=df.index)
    swing_low = pd.Series(False, index=df.index)

    for i in range(SwingPeriod, len(df) - SwingPeriod):
        if high.iloc[i] >= pivot_high.iloc[i] - 1e-10:
            swing_high.iloc[i] = True
        if low.iloc[i] <= pivot_low.iloc[i] + 1e-10:
            swing_low.iloc[i] = True

    bull_fvg = pd.Series(False, index=df.index)
    bear_fvg = pd.Series(False, index=df.index)
    bull_fvg_mid = pd.Series(np.nan, index=df.index)
    bear_fvg_mid = pd.Series(np.nan, index=df.index)

    for i in range(2, len(df)):
        bull_cond = (low.iloc[i] > high.iloc[i-1]) and (low.iloc[i-2] > high.iloc[i-1])
        bear_cond = (high.iloc[i] < low.iloc[i-1]) and (high.iloc[i-2] < low.iloc[i-1])
        if bull_cond:
            bull_fvg.iloc[i] = True
            bull_fvg_mid.iloc[i] = (low.iloc[i] + high.iloc[i-1]) / 2
        if bear_cond:
            bear_fvg.iloc[i] = True
            bear_fvg_mid.iloc[i] = (high.iloc[i] + low.iloc[i-1]) / 2

    h_indexes = []
    h_prices = []
    l_indexes = []
    l_prices = []

    for i in range(len(df)):
        if swing_high.iloc[i]:
            h_indexes.append(i)
            h_prices.append(pivot_high.iloc[i])
        if swing_low.iloc[i]:
            l_indexes.append(i)
            l_prices.append(pivot_low.iloc[i])

    mss_long = pd.Series(False, index=df.index)
    mss_short = pd.Series(False, index=df.index)

    for i in range(SwingPeriod, len(df)):
        for j in range(len(h_indexes) - 1, -1, -1):
            if h_indexes[j] < i and h_indexes[j] > i - 1000:
                level = h_prices[j]
                if close.iloc[i] > level + atr.iloc[i] * ATR_Coefficient:
                    mss_long.iloc[i] = True
                    break
                break

    for i in range(SwingPeriod, len(df)):
        for j in range(len(l_indexes) - 1, -1, -1):
            if l_indexes[j] < i and l_indexes[j] > i - 1000:
                level = l_prices[j]
                if close.iloc[i] < level - atr.iloc[i] * ATR_Coefficient:
                    mss_short.iloc[i] = True
                    break
                break

    entry_signal = pd.Series(False, index=df.index)
    entry_direction = pd.Series(dtype=str, index=df.index)

    for i in range(SwingPeriod, len(df)):
        if mss_long.iloc[i] and not bull_fvg.iloc[i]:
            entry_signal.iloc[i] = True
            entry_direction.iloc[i] = 'long'
        elif mss_short.iloc[i] and not bear_fvg.iloc[i]:
            entry_signal.iloc[i] = True
            entry_direction.iloc[i] = 'short'

    entries = []
    trade_num = 1

    for i in range(SwingPeriod, len(df)):
        if entry_signal.iloc[i] and not entry_signal.iloc[i-1] if i > 0 else True:
            direction = entry_direction.iloc[i]
            ts = int(df['time'].iloc[i])
            et = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            price = float(close.iloc[i])

            entry = {
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': et,
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            }
            entries.append(entry)
            trade_num += 1

    return entries