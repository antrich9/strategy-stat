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
    results = []
    trade_num = 1

    close = df['close']
    open_vals = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    time = df['time']

    n = len(df)

    # Input parameters
    lookback_bars = 12
    threshold = 0.0

    # FVG conditions
    bull_fvg = (low > high.shift(2)) & (close.shift(1) > high.shift(2))
    bear_fvg = (high < low.shift(2)) & (close.shift(1) < low.shift(2))

    # OB conditions
    is_up = close > open_vals
    is_down = close < open_vals

    is_ob_up = is_down.shift(1) & is_up & (close > high.shift(1))
    is_ob_down = is_up.shift(1) & is_down & (close < low.shift(1))

    # Bars since functions
    def barssince_long(condition_series):
        result = pd.Series(np.nan, index=condition_series.index)
        count = 0
        for i in range(len(condition_series)):
            if condition_series.iloc[i]:
                count = 0
            else:
                count += 1
            result.iloc[i] = count
        return result

    def barssince_short(condition_series):
        result = pd.Series(np.nan, index=condition_series.index)
        count = 0
        for i in range(len(condition_series)):
            if condition_series.iloc[i]:
                count = 0
            else:
                count += 1
            result.iloc[i] = count
        return result

    bull_since = barssince_long(bull_fvg.fillna(False))
    bear_since = barssince_short(bear_fvg.fillna(False))

    # Bullish BPR
    bull_cond_1 = bull_fvg & (bull_since <= lookback_bars)
    combined_low_bull = pd.Series(np.nan, index=df.index)
    combined_high_bull = pd.Series(np.nan, index=df.index)
    bull_result = pd.Series(False, index=df.index)

    for i in range(2, n):
        if bull_cond_1.iloc[i]:
            bs = int(bull_since.iloc[i])
            if bs >= 0 and bs + 2 < n:
                combined_low_bull.iloc[i] = max(high.iloc[i - bs], high.iloc[i - 2])
                combined_high_bull.iloc[i] = min(low.iloc[i - bs + 2], low.iloc[i])
                if not np.isnan(combined_high_bull.iloc[i]) and not np.isnan(combined_low_bull.iloc[i]):
                    bull_result.iloc[i] = (combined_high_bull.iloc[i] - combined_low_bull.iloc[i]) >= threshold

    # Bearish BPR
    bear_cond_1 = bear_fvg & (bear_since <= lookback_bars)
    combined_low_bear = pd.Series(np.nan, index=df.index)
    combined_high_bear = pd.Series(np.nan, index=df.index)
    bear_result = pd.Series(False, index=df.index)

    for i in range(2, n):
        if bear_cond_1.iloc[i]:
            bs = int(bear_since.iloc[i])
            if bs >= 0 and bs + 2 < n:
                combined_low_bear.iloc[i] = max(high.iloc[i - bs + 2], high.iloc[i])
                combined_high_bear.iloc[i] = min(low.iloc[i - bs], low.iloc[i - 2])
                if not np.isnan(combined_high_bear.iloc[i]) and not np.isnan(combined_low_bear.iloc[i]):
                    bear_result.iloc[i] = (combined_high_bear.iloc[i] - combined_low_bear.iloc[i]) >= threshold

    # Entry conditions based on bull_result and bear_result
    long_condition = bull_result.fillna(False)
    short_condition = bear_result.fillna(False)

    for i in range(n):
        entry_price = close.iloc[i]
        ts = int(time.iloc[i])
        entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        if long_condition.iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        if short_condition.iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return results