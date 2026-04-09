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
    swingLength = 5
    htf1 = '240min'
    htf2 = '60min'

    def calc_trend(series_high, series_low, series_close, swing_len):
        swing_high = series_high.rolling(window=swing_len, min_periods=swing_len).max()
        swing_low = series_low.rolling(window=swing_len, min_periods=swing_len).min()
        swing_high_prev = swing_high.shift(1)
        swing_low_prev = swing_low.shift(1)
        trend = pd.Series(0, index=series_close.index)
        valid_mask = swing_high_prev.notna() & swing_low_prev.notna()
        trend.loc[valid_mask] = np.where(
            series_close.loc[valid_mask] > swing_high_prev.loc[valid_mask], 1,
            np.where(series_close.loc[valid_mask] < swing_low_prev.loc[valid_mask], -1, 0)
        )
        trend = trend.replace(0, np.nan).ffill().fillna(0)
        return trend

    htf1_df = df.set_index('time')
    htf1_df.index = pd.to_datetime(htf1_df.index, unit='s', utc=True)
    htf1_resampled = htf1_df.resample(htf1, origin='start_time', label='right', closed='right').agg({'high': 'max', 'low': 'min', 'close': 'last'})
    htf1_trend = calc_trend(htf1_resampled['high'], htf1_resampled['low'], htf1_resampled['close'], swingLength)
    htf1_trend.name = 'trendHTF1'
    htf1_merged = htf1_resampled.join(htf1_trend).reindex(htf1_df.index).ffill()['trendHTF1']

    htf2_df = df.set_index('time')
    htf2_df.index = pd.to_datetime(htf2_df.index, unit='s', utc=True)
    htf2_resampled = htf2_df.resample(htf2, origin='start_time', label='right', closed='right').agg({'high': 'max', 'low': 'min', 'close': 'last'})
    htf2_trend = calc_trend(htf2_resampled['high'], htf2_resampled['low'], htf2_resampled['close'], swingLength)
    htf2_trend.name = 'trendHTF2'
    htf2_merged = htf2_resampled.join(htf2_trend).reindex(htf2_df.index).ffill()['trendHTF2']

    longCondition = (htf1_merged == 1) & (htf2_merged == 1)
    shortCondition = (htf1_merged == -1) & (htf2_merged == -1)

    entries = []
    trade_num = 0
    in_position = False

    for i in range(len(df)):
        ts = int(df['time'].iloc[i])
        if i < swingLength:
            continue
        if in_position:
            continue
        if longCondition.iloc[i]:
            trade_num += 1
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            in_position = True
        elif shortCondition.iloc[i]:
            trade_num += 1
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            in_position = True

    return entries