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
    atrPeriod = 14
    volumeThreshold = 1.0
    rviPeriod = 10
    baselinePeriod = 20
    ashPeriod = 14

    close = df['close']
    high = df['high']
    low = df['low']
    open_col = df['open']
    volume = df['volume']

    # 1. ATR (Wilder)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atrPeriod, adjust=False).mean()

    # 2. Baseline (EMA)
    baseline = close.ewm(span=baselinePeriod, adjust=False).mean()

    # 3. ASH (RSI Wilder)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.ewm(alpha=1/ashPeriod, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/ashPeriod, adjust=False).mean()
    rs = avg_gain / avg_loss
    ash = 100 - (100 / (1 + rs))

    # 4. SuperTrend (multiplier=3, atr_len=10)
    st_multiplier = 3
    st_atr_len = 10
    st_tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    st_atr = st_tr.ewm(alpha=1/st_atr_len, adjust=False).mean()
    st_upperband = (high + low) / 2 + st_multiplier * st_atr
    st_lowerband = (high + low) / 2 - st_multiplier * st_atr

    supertrend = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1, index=df.index)  # 1=long(downtrend), -1=short(uptrend)

    prev_st = pd.Series(np.nan, index=df.index)
    prev_dir = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        if i == 1:
            supertrend.iloc[i] = st_upperband.iloc[i]
            direction.iloc[i] = 1
        else:
            if st_upperband.iloc[i] < prev_st.iloc[i-1] or close.iloc[i-1] > prev_st.iloc[i-1]:
                supertrend.iloc[i] = st_upperband.iloc[i]
            else:
                supertrend.iloc[i] = prev_st.iloc[i-1]

            if st_lowerband.iloc[i] > prev_st.iloc[i-1] or close.iloc[i-1] < prev_st.iloc[i-1]:
                supertrend.iloc[i] = st_lowerband.iloc[i]

            if close.iloc[i] > supertrend.iloc[i]:
                direction.iloc[i] = -1
            elif close.iloc[i] < supertrend.iloc[i]:
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = prev_dir.iloc[i-1]

        prev_st.iloc[i] = supertrend.iloc[i]
        prev_dir.iloc[i] = direction.iloc[i]

    # 5. Volume filter
    volumeMA = volume.rolling(20).mean()
    isVolumeValid = volume > volumeMA * volumeThreshold

    # 6. RVI
    num = close - open_col
    den = high - low
    num_sum = num.rolling(rviPeriod).sum()
    den_sum = den.rolling(rviPeriod).sum()
    num_ma = num_sum.rolling(4).mean()
    den_ma = den_sum.rolling(4).mean()
    rvi = pd.Series(np.where(den_ma != 0, num_ma / den_ma, 0), index=df.index)
    rviSignal = rvi.rolling(4).mean()

    # Entry conditions
    close_above_baseline_prev = close.shift(1) <= baseline.shift(1)
    close_above_baseline = close > baseline
    longCondition = (close_above_baseline & close_above_baseline_prev) & (ash > 50) & (direction < 0) & isVolumeValid

    close_below_baseline_prev = close.shift(1) >= baseline.shift(1)
    close_below_baseline = close < baseline
    shortCondition = (close_below_baseline & close_below_baseline_prev) & (ash < 50) & (direction > 0) & isVolumeValid

    # Find valid bars (no NaN in key indicators)
    valid_bars = ~(atr.isna() | baseline.isna() | ash.isna() | supertrend.isna() | volumeMA.isna() | rvi.isna())

    entries = []
    trade_num = 1
    in_position = False

    for i in range(len(df)):
        if not valid_bars.iloc[i]:
            continue

        if not in_position:
            if longCondition.iloc[i]:
                ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
                trade_num += 1
                in_position = True
            elif shortCondition.iloc[i]:
                ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
                trade_num += 1
                in_position = True
        else:
            # Reset position state when flat (for next iteration check)
            pass

    return entries