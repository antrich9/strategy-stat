import pandas as pd
import numpy as np
from datetime import datetime, timezone

def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def _is_up(idx: int, close: pd.Series, open_col: pd.Series) -> bool:
    return close.iloc[idx] > open_col.iloc[idx]

def _is_down(idx: int, close: pd.Series, open_col: pd.Series) -> bool:
    return close.iloc[idx] < open_col.iloc[idx]

def _is_ob_up(idx: int, close: pd.Series, open_col: pd.Series, high: pd.Series, low: pd.Series) -> bool:
    return _is_down(idx + 1, close, open_col) and _is_up(idx, close, open_col) and close.iloc[idx] > high.iloc[idx + 1]

def _is_ob_down(idx: int, close: pd.Series, open_col: pd.Series, high: pd.Series, low: pd.Series) -> bool:
    return _is_up(idx + 1, close, open_col) and _is_down(idx, close, open_col) and close.iloc[idx] < low.iloc[idx + 1]

def _is_fvg_up(idx: int, low: pd.Series, high: pd.Series) -> bool:
    return low.iloc[idx] > high.iloc[idx + 2]

def _is_fvg_down(idx: int, low: pd.Series, high: pd.Series) -> bool:
    return high.iloc[idx] < low.iloc[idx + 2]

def generate_entries(df: pd.DataFrame) -> list:
    fastLength = 50
    slowLength = 200
    period = 14

    fastEMA = df['close'].ewm(span=fastLength, adjust=False).mean()
    slowEMA = df['close'].ewm(span=slowLength, adjust=False).mean()

    timestamps = pd.to_datetime(df['time'], unit='s', utc=True)
    df_with_ts = df.copy()
    df_with_ts['ts'] = timestamps
    df_with_ts['day'] = df_with_ts['ts'].dt.date

    daily_agg = df_with_ts.groupby('day').agg({
        'high': 'max',
        'low': 'min',
        'time': 'min'
    }).shift(1)
    daily_agg = daily_agg.rename(columns={'high': 'prev_day_high', 'low': 'prev_day_low', 'time': 'prev_day_time'})

    df_with_ts = df_with_ts.merge(daily_agg[['prev_day_high', 'prev_day_low']], left_on='day', right_index=True, how='left')
    prevDayHigh = df_with_ts['prev_day_high']
    prevDayLow = df_with_ts['prev_day_low']

    flagpdh = (df['close'] > prevDayHigh)
    flagpdl = (df['close'] < prevDayLow)

    def is_time_in_session(ts: int, start_h: int, start_m: int, end_h: int, end_m: int) -> bool:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        cur_min = dt.hour * 60 + dt.minute
        start_min = start_h * 60 + start_m
        end_min = end_h * 60 + end_m
        return start_min <= cur_min <= end_min

    time_cond1 = df['time'].apply(lambda t: is_time_in_session(t, 7, 0, 9, 59))
    time_cond2 = df['time'].apply(lambda t: is_time_in_session(t, 12, 0, 14, 59))
    is_time = time_cond1 | time_cond2

    ob_up_series = pd.Series(False, index=df.index)
    ob_down_series = pd.Series(False, index=df.index)
    fvg_up_series = pd.Series(False, index=df.index)
    fvg_down_series = pd.Series(False, index=df.index)

    for i in range(2, len(df)):
        ob_up_series.iloc[i] = _is_ob_up(1, df['close'], df['open'], df['high'], df['low'])
        ob_down_series.iloc[i] = _is_ob_down(1, df['close'], df['open'], df['high'], df['low'])
        fvg_up_series.iloc[i] = _is_fvg_up(0, df['low'], df['high'])
        fvg_down_series.iloc[i] = _is_fvg_down(0, df['low'], df['high'])

    stacked_bull = ob_up_series & fvg_up_series
    stacked_bear = ob_down_series & fvg_down_series

    min_bar = max(slowLength, 2)
    valid_bars = fastEMA.notna() & slowEMA.notna()

    crossOver = (fastEMA > slowEMA) & (fastEMA.shift(1) <= slowEMA.shift(1))
    crossUnder = (fastEMA < slowEMA) & (fastEMA.shift(1) >= slowEMA.shift(1))

    long_cond = crossOver & flagpdh & is_time & stacked_bull & valid_bars
    short_cond = crossUnder & flagpdl & is_time & stacked_bear & valid_bars

    entries = []
    trade_num = 1

    for i in range(min_bar, len(df)):
        if not valid_bars.iloc[i]:
            continue
        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            }
            entries.append(entry)
            trade_num += 1
        elif short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            }
            entries.append(entry)
            trade_num += 1

    return entries