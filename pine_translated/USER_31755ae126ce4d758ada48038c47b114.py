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
    # ---- T3 Indicator ----
    t3_len = 100
    b = 0.7
    c1 = -b ** 3
    c2 = 3 * b ** 2 + 3 * b ** 3
    c3 = -6 * b ** 2 - 3 * b - 3 * b ** 3
    c4 = 1 + 3 * b + b ** 3 + 3 * b ** 2

    e1 = df['close'].ewm(span=t3_len, adjust=False).mean()
    e2 = e1.ewm(span=t3_len, adjust=False).mean()
    e3 = e2.ewm(span=t3_len, adjust=False).mean()
    e4 = e3.ewm(span=t3_len, adjust=False).mean()
    e5 = e4.ewm(span=t3_len, adjust=False).mean()
    e6 = e5.ewm(span=t3_len, adjust=False).mean()
    T3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    # ---- Chaikin Volatility ----
    chaikin_len = 10
    chaikin_thresh = 0.5
    high_low = df['high'] - df['low']
    chaikin_ema = high_low.ewm(span=chaikin_len, adjust=False).mean()
    chaikin_vol = chaikin_ema.diff()
    is_volatile = chaikin_vol > chaikin_thresh

    # ---- TDFI Indicator ----
    lookback = 13
    mma_len = 13
    smma_len = 13
    n_len = 3
    filter_high = 0.05
    filter_low = -0.05

    # simple EMA helper (only EMA mode is required for default settings)
    def ma_ema(src, length):
        return src.ewm(span=length, adjust=False).mean()

    price = df['close']
    mma = ma_ema(price * 1000, mma_len)
    smma = ma_ema(mma, smma_len)

    impet_mma = mma - mma.shift(1)
    impet_smma = smma - smma.shift(1)
    div_ma = np.abs(mma - smma)
    aver_impet = (impet_mma + impet_smma) / 2
    tdf = div_ma * (aver_impet ** n_len)

    roll_max = tdf.abs().rolling(window=lookback * n_len, min_periods=1).max()
    # avoid division by zero
    roll_max = roll_max.replace(0, np.nan)
    signal = tdf / roll_max
    signal = signal.fillna(0)

    # ---- Crossover / Crossunder helpers ----
    def crossover(series, level):
        return (series > level) & (series.shift(1) <= level)

    def crossunder(series, level):
        return (series < level) & (series.shift(1) >= level)

    # TDFI signals
    signal_long_tdfi = crossover(signal, filter_high)
    signal_short_tdfi = crossunder(signal, filter_low)

    # Apply inverse (inverseTDFI = true) → long uses short signal, short uses long signal
    final_long_signal = signal_short_tdfi
    final_short_signal = signal_long_tdfi

    # ---- Entry Conditions ----
    long_condition = (df['close'] > T3) & final_long_signal & is_volatile
    short_condition = (df['close'] < T3) & final_short_signal & is_volatile

    # ---- Generate Entry List ----
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries