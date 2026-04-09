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
    lookback_bars = 12

    high = df['high']
    low = df['low']
    close = df['close']
    open_vals = df['open']
    ts = df['time']

    bull_fvg1 = (high.shift(1) < low.shift(3)) & (close.shift(2) < low.shift(3))

    bear_fvg1 = (high.shift(3) > low.shift(1)) & (close.shift(2) > low.shift(1))

    bull_since = pd.Series(np.nan, index=df.index)
    bull_since = bull_since.where(bull_fvg1, pd.concat([bull_since.shift(1) + 1, pd.Series([0], index=bull_fvg1[bull_fvg1].index)]).reindex(df.index).ffill())
    bull_since = bull_since.where(bull_fvg1 | bull_since.shift(1).notna(), np.nan)

    bear_since = pd.Series(np.nan, index=df.index)
    bear_since = bear_since.where(bear_fvg1, pd.concat([bear_since.shift(1) + 1, pd.Series([0], index=bear_fvg1[bear_fvg1].index)]).reindex(df.index).ffill())
    bear_since = bear_since.where(bear_fvg1 | bear_since.shift(1).notna(), np.nan)

    bull_cond_1 = bull_fvg1 & (bull_since <= lookback_bars)

    combined_low_bull = np.where(bull_cond_1, np.maximum(high.shift(bull_since.astype(int)), high.shift(3)), np.nan)
    combined_high_bull = np.where(bull_cond_1, np.minimum(low.shift(bull_since.astype(int) + 3), low.shift(1)), np.nan)
    bull_result = bull_cond_1 & (combined_high_bull - combined_low_bull >= 0)

    bear_cond_1 = bear_fvg1 & (bear_since <= lookback_bars)
    combined_low_bear = np.where(bear_cond_1, np.maximum(high.shift(bear_since.astype(int) + 3), high.shift(1)), np.nan)
    combined_high_bear = np.where(bear_cond_1, np.minimum(low.shift(bear_since.astype(int)), low.shift(3)), np.nan)
    bear_result = bear_cond_1 & (combined_high_bear - combined_low_bear >= 0)

    london_times = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    london_hour = london_times.hour
    london_minute = london_times.hour * 60 + london_times.minute
    in_trading_window = ((london_hour == 8) & (london_minute >= 465) & (london_minute < 585)) | \
                        ((london_hour == 15) & (london_minute >= 885) & (london_minute < 1005))

    swing_alignment_bullish = pd.Series(True, index=df.index)
    swing_alignment_bearish = pd.Series(True, index=df.index)

    confirmed_long_condition = bull_result & in_trading_window & swing_alignment_bullish
    confirmed_short_condition = bear_result & in_trading_window & swing_alignment_bearish

    entries = []
    trade_num = 1

    for i in range(4, len(df)):
        if confirmed_long_condition.iloc[i]:
            entry_price = close.iloc[i - 1]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    for i in range(4, len(df)):
        if confirmed_short_condition.iloc[i]:
            entry_price = close.iloc[i - 1]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries