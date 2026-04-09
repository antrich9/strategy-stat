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
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']

    lengthT3 = 5
    factorT3 = 0.7
    ff_up_tolerance = 2.5
    ff_down_tolerance = 2.5
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    adx_len = 14
    di_len = 14
    maxConcurrentTrades = 5

    ema1 = close.ewm(span=lengthT3, adjust=False).mean()
    ema2 = ema1.ewm(span=lengthT3, adjust=False).mean()
    ema3 = ema2.ewm(span=lengthT3, adjust=False).mean()
    ema4 = ema3.ewm(span=lengthT3, adjust=False).mean()
    ema5 = ema4.ewm(span=lengthT3, adjust=False).mean()
    ema6 = ema5.ewm(span=lengthT3, adjust=False).mean()
    c1 = factorT3
    c2 = factorT3 * factorT3
    c3 = factorT3 * c2
    c4 = -factorT3 * (1 + c1 + c2)
    t3 = c1 * ema6 + c2 * ema5 + c3 * ema4 + c4 * ema3

    t3_signals = (t3 > t3.shift(1)).astype(int) - (t3 < t3.shift(1)).astype(int)
    basic_long = t3_signals > 0
    basic_short = t3_signals < 0

    trendilo_pct_change = close.diff(trendilo_smooth) / close * 100
    window = trendilo_length
    m = int(trendilo_offset * (window - 1))
    s = (window - 1) / 6.0
    k = np.arange(window)
    weights = np.exp(-((k - m) ** 2) / (2 * s ** 2))
    weights = weights / weights.sum()
    avg_pct_change = trendilo_pct_change.rolling(window).apply(lambda x: np.sum(weights * x.values), raw=True)
    rms = trendilo_bmult * np.sqrt((avg_pct_change ** 2).rolling(trendilo_length).mean())

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    prev_open = open_price.shift(1)
    bar_upper = np.where(prev_close >= prev_open, prev_close, prev_open)
    bar_lower = np.where(prev_close <= prev_open, prev_close, prev_open)
    ff_up = np.where(high > bar_upper, 100 * (high - bar_upper) / bar_upper, 0)
    ff_down = np.where(low < bar_lower, 100 * (bar_lower - low) / bar_lower, 0)
    ff_up_signal = ff_up >= ff_up_tolerance
    ff_down_signal = ff_down >= ff_down_tolerance

    up_move = high - prev_high
    down_move = prev_low - low
    plusDM = ((up_move > down_move) & (up_move > 0)) * up_move
    minusDM = ((down_move > up_move) & (down_move > 0)) * down_move
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    tr_s = pd.Series(tr).ewm(alpha=1.0/di_len, adjust=False).mean()
    plusDI = 100 * pd.Series(plusDM.values).ewm(alpha=1.0/di_len, adjust=False).mean() / tr_s
    minusDI = 100 * pd.Series(minusDM.values).ewm(alpha=1.0/di_len, adjust=False).mean() / tr_s
    plusDI = plusDI.fillna(0)
    minusDI = minusDI.fillna(0)
    dx = 100 * np.abs(plusDI - minusDI) / (plusDI + minusDI + 1e-10)
    adx_value = pd.Series(dx).ewm(alpha=1.0/adx_len, adjust=False).mean()

    long_condition = basic_long & (trendilo_dir == 1) & (adx_value > 20) & (~ff_down_signal)
    short_condition = basic_short & (trendilo_dir == -1) & (adx_value > 20) & (~ff_up_signal)

    trendilo_dir = pd.Series(np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0)), index=df.index)

    entries = []
    trade_num = 1
    trade_done = False
    last_trendilo_dir = None
    active_trades = 0

    for i in range(1, len(df)):
        if not pd.isna(trendilo_dir.iloc[i]):
            if last_trendilo_dir is not None and trendilo_dir.iloc[i] != last_trendilo_dir:
                trade_done = False
                active_trades = 0
            last_trendilo_dir = trendilo_dir.iloc[i]

        if long_condition.iloc[i] and not trade_done and active_trades < maxConcurrentTrades:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            trade_done = True
            active_trades += 1
        elif short_condition.iloc[i] and not trade_done and active_trades < maxConcurrentTrades:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            trade_done = True
            active_trades += 1

    return entries