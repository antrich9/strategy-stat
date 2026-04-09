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
    # Keep a copy and preserve original timestamps
    df = df.copy()
    df['ts'] = df['time'].values

    # Convert time to datetime (UTC) for time‑based filters
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('dt')

    # ------------------------------------------------------------------
    # 4H (240‑minute) higher‑timeframe data
    # ------------------------------------------------------------------
    high_4h = df['high'].resample('240T').max()
    low_4h  = df['low'].resample('240T').min()
    close_4h = df['close'].resample('240T').last()

    # 4H FVG needs the 4H bar two periods ago
    high_4h_2 = high_4h.shift(2)
    low_4h_2  = low_4h.shift(2)

    # Align 4H values back to the original 15‑minute bars (forward‑fill)
    df['high_4h']  = high_4h.reindex(df.index, method='ffill')
    df['low_4h']   = low_4h.reindex(df.index, method='ffill')
    df['high_4h_2'] = high_4h_2.reindex(df.index, method='ffill')
    df['low_4h_2']  = low_4h_2.reindex(df.index, method='ffill')
    df['close_4h'] = close_4h.reindex(df.index, method='ffill')

    # ------------------------------------------------------------------
    # Identify the last bar of each 4H period (mimics barstate.isconfirmed)
    # ------------------------------------------------------------------
    period_4h = df.index.to_period('240T')
    is_last_4h_bar = period_4h != period_4h.shift(-1)   # True on the final 15‑min bar of each 4H candle

    # ------------------------------------------------------------------
    # Trading windows (London session: 07:45‑09:45 and 14:45‑16:45)
    # ------------------------------------------------------------------
    hour = df.index.hour
    minute = df.index.minute
    in_window1 = ((hour == 7) & (minute >= 45)) | (hour == 8) | ((hour == 9) & (minute <= 45))
    in_window2 = ((hour == 14) & (minute >= 45)) | (hour == 15) | ((hour == 16) & (minute <= 45))
    df['in_trading_window'] = in_window1 | in_window2

    # ------------------------------------------------------------------
    # 15‑minute order‑block and fair‑value‑gap conditions
    # ------------------------------------------------------------------
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']

    isObUp   = (close > open_) & (close.shift(1) < open_.shift(1)) & (close > high.shift(1))
    isObDown = (close < open_) & (close.shift(1) > open_.shift(1)) & (close < low.shift(1))
    isFvgUp   = low > high.shift(2)
    isFvgDown = high < low.shift(2)

    bullishSignal = isObUp & isFvgUp
    bearishSignal = isObDown & isFvgDown

    # ------------------------------------------------------------------
    # 4H FVG (confirmed only on the last 15‑min bar of the 4H period)
    # ------------------------------------------------------------------
    bullishFVG_4h = is_last_4h_bar & (df['low_4h'] > df['high_4h_2'])
    bearishFVG_4h = is_last_4h_bar & (df['high_4h'] < df['low_4h_2'])

    # ------------------------------------------------------------------
    # Entry signals
    # ------------------------------------------------------------------
    long_cond  = bullishFVG_4h & bullishSignal & df['in_trading_window']
    short_cond = bearishFVG_4h & bearishSignal & df['in_trading_window']

    # ------------------------------------------------------------------
    # Build entry list
    # ------------------------------------------------------------------
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if long_cond.iloc[i]:
            ts = int(df['ts'].iloc[i])
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
        elif short_cond.iloc[i]:
            ts = int(df['ts'].iloc[i])
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