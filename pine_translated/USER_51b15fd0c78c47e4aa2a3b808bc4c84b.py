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
    trade_num = 0

    if len(df) < 3:
        return entries

    open_col = df['open']
    high_col = df['high']
    low_col = df['low']
    close_col = df['close']

    bull_fvg = (low_col > high_col.shift(2)) & (close_col.shift(1) > high_col.shift(2))
    bear_fvg = (high_col < low_col.shift(2)) & (close_col.shift(1) < low_col.shift(2))

    bull_since = np.where(bull_fvg, np.arange(len(df)), 0)
    for i in range(1, len(df)):
        if not bull_fvg.iloc[i]:
            bull_since[i] = bull_since[i-1] + 1 if i > 0 else 0
    bull_since = pd.Series(bull_since, index=df.index)

    lookback_bars = 12
    threshold = 0.0

    bull_cond_1 = bull_fvg & (bull_since <= lookback_bars)

    combined_low_bull = pd.Series(np.nan, index=df.index)
    combined_high_bull = pd.Series(np.nan, index=df.index)
    bull_result = pd.Series(False, index=df.index)

    for i in range(3, len(df)):
        if bull_cond_1.iloc[i]:
            bs = int(bull_since.iloc[i])
            combined_low_bull.iloc[i] = max(high_col.iloc[bs] if bs < len(df) else np.nan, high_col.iloc[2] if 2 < len(df) else np.nan)
            combined_high_bull.iloc[i] = min(low_col.iloc[bs + 2] if bs + 2 < len(df) else np.nan, low_col.iloc[i] if i < len(df) else np.nan)
            if not np.isnan(combined_low_bull.iloc[i]) and not np.isnan(combined_high_bull.iloc[i]):
                bull_result.iloc[i] = (combined_high_bull.iloc[i] - combined_low_bull.iloc[i]) >= threshold

    bear_since = np.where(bear_fvg, np.arange(len(df)), 0)
    for i in range(1, len(df)):
        if not bear_fvg.iloc[i]:
            bear_since[i] = bear_since[i-1] + 1 if i > 0 else 0
    bear_since = pd.Series(bear_since, index=df.index)

    bear_cond_1 = bear_fvg & (bear_since <= lookback_bars)

    combined_low_bear = pd.Series(np.nan, index=df.index)
    combined_high_bear = pd.Series(np.nan, index=df.index)
    bear_result = pd.Series(False, index=df.index)

    for i in range(3, len(df)):
        if bear_cond_1.iloc[i]:
            bs = int(bear_since.iloc[i])
            combined_low_bear.iloc[i] = max(high_col.iloc[i] if i < len(df) else np.nan, high_col.iloc[2] if 2 < len(df) else np.nan)
            combined_high_bear.iloc[i] = min(low_col.iloc[bs] if bs < len(df) else np.nan, low_col.iloc[2] if 2 < len(df) else np.nan)
            if not np.isnan(combined_low_bear.iloc[i]) and not np.isnan(combined_high_bear.iloc[i]):
                bear_result.iloc[i] = (combined_high_bear.iloc[i] - combined_low_bear.iloc[i]) >= threshold

    bull_fvg_plot = low_col > high_col.shift(2)
    bear_fvg_plot = high_col < low_col.shift(2)

    TopImbalance_Bway = (low_col.shift(2) <= open_col.shift(1)) & (high_col >= close_col.shift(1)) & (close_col < low_col.shift(1))
    Top_ImbXBway = (low_col.shift(2) <= open_col.shift(1)) & (high_col >= close_col.shift(1)) & (close_col > low_col.shift(1))
    TopImbalancesize = low_col.shift(2) - high_col

    BottomInbalance_Bway = (high_col.shift(2) >= open_col.shift(1)) & (low_col <= close_col.shift(1)) & (close_col > high_col.shift(1))
    Bottom_ImbXBAway = (high_col.shift(2) >= open_col.shift(1)) & (low_col <= close_col.shift(1)) & (close_col < high_col.shift(1))
    BottomInbalancesize = low_col - high_col.shift(2)

    long_condition = bull_result
    short_condition = bear_result

    for i in range(len(df)):
        if long_condition.iloc[i]:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

    for i in range(len(df)):
        if short_condition.iloc[i]:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

    return entries