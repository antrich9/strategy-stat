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
    trade_num = 1

    df = df.copy()
    df['time'] = df['time'].astype(int)

    def to_london_ts(ts, hour, minute):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return int(datetime(dt.year, dt.month, dt.day, hour, minute, tzinfo=timezone.utc).timestamp())

    london_start_morning = df['time'].apply(lambda x: to_london_ts(x, 7, 45))
    london_end_morning = df['time'].apply(lambda x: to_london_ts(x, 9, 45))
    london_start_afternoon = df['time'].apply(lambda x: to_london_ts(x, 14, 45))
    london_end_afternoon = df['time'].apply(lambda x: to_london_ts(x, 16, 45))

    in_trading_window = ((df['time'] >= london_start_morning) & (df['time'] < london_end_morning)) | \
                        ((df['time'] >= london_start_afternoon) & (df['time'] < london_end_afternoon))

    ob_up = (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open']) & (df['close'] > df['high'].shift(1))
    ob_down = (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open']) & (df['close'] < df['low'].shift(1))
    fvg_up = df['low'] > df['high'].shift(2)
    fvg_down = df['high'] < df['low'].shift(2)

    vol_filt = df['volume'].shift(1) > df['volume'].ewm(span=9, adjust=False).mean() * 1.5

    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atr_filt = (df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr)

    loc = df['close'].ewm(span=54, adjust=False).mean()
    loc_trend = loc > loc.shift(1)

    stacked_long = ob_up & fvg_up
    stacked_short = ob_down & fvg_down

    long_cond = stacked_long.fillna(False) & vol_filt.fillna(False) & atr_filt.fillna(False) & loc_trend.fillna(False)
    short_cond = stacked_short.fillna(False) & vol_filt.fillna(False) & atr_filt.fillna(False) & (~loc_trend).fillna(False)

    long_signal = long_cond & (~long_cond.shift(1).fillna(False))
    short_signal = short_cond & (~short_cond.shift(1).fillna(False))

    for i in range(1, len(df)):
        if long_signal.iloc[i] and in_trading_window.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
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
            })
            trade_num += 1
        elif short_signal.iloc[i] and in_trading_window.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
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
            })
            trade_num += 1

    return entries