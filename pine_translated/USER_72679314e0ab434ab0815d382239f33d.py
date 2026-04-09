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
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('timestamp')

    # Resample to 4H
    ohlc_4h = df[['open', 'high', 'low', 'close', 'volume']].resample('240min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Volume Filter
    vol_sma_4h = ohlc_4h['volume'].rolling(9).mean() * 1.5
    vol_filt_4h = ohlc_4h['volume'].shift(1) > vol_sma_4h

    # ATR Filter (Wilder)
    high_arr = ohlc_4h['high'].values
    low_arr = ohlc_4h['low'].values
    close_arr = ohlc_4h['close'].values
    tr = np.maximum(high_arr[1:] - low_arr[1:], np.maximum(np.abs(high_arr[1:] - close_arr[:-1]), np.abs(low_arr[1:] - close_arr[:-1])))
    atr = np.zeros(len(ohlc_4h))
    atr[0] = tr[:14].mean() if len(tr) >= 14 else tr.mean()
    for i in range(1, len(atr)):
        atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    atr_4h = pd.Series(atr, index=ohlc_4h.index) / 1.5

    high_4h_series = ohlc_4h['high']
    low_4h_series = ohlc_4h['low']
    close_4h_series = ohlc_4h['close']

    atr_filt_4h = ((low_4h_series - high_4h_series.shift(2) > atr_4h) | (low_4h_series.shift(2) - high_4h_series > atr_4h))

    # Trend Filter
    loc = close_4h_series.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    loc_filt_b_4h = loc2
    loc_filt_s_4h = ~loc2

    # Bullish/Bearish FVGs on 4H
    bfvg = low_4h_series > high_4h_series.shift(2)
    sfvg = high_4h_series < low_4h_series.shift(2)

    bull_fvg_cond = bfvg & vol_filt_4h & atr_filt_4h & loc_filt_b_4h
    bear_fvg_cond = sfvg & vol_filt_4h & atr_filt_4h & loc_filt_s_4h

    entries = []
    trade_num = 1
    last_fvg = 0

    for i in range(1, len(ohlc_4h)):
        if i < 3:
            continue

        curr_ts = ohlc_4h.index[i].value // 10**6
        curr_price = close_4h_series.iloc[i]

        # Sharp turn detection
        if bull_fvg_cond.iloc[i] and last_fvg == -1:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': curr_ts,
                'entry_time': datetime.fromtimestamp(curr_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': curr_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': curr_price,
                'raw_price_b': curr_price
            })
            trade_num += 1
            last_fvg = 1
        elif bear_fvg_cond.iloc[i] and last_fvg == 1:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': curr_ts,
                'entry_time': datetime.fromtimestamp(curr_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': curr_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': curr_price,
                'raw_price_b': curr_price
            })
            trade_num += 1
            last_fvg = -1
        elif bull_fvg_cond.iloc[i]:
            last_fvg = 1
        elif bear_fvg_cond.iloc[i]:
            last_fvg = -1

    return entries