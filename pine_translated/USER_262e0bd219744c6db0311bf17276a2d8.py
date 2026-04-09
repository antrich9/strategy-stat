import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    time = df['time']

    # SuperTrend Parameters
    st_len = 10
    st_mult = 3.0

    # ADX Parameters
    adx_len = 14
    adx_thresh = 25

    # Squeeze Parameters
    squeeze_len = 20
    kc_mult = 1.5
    bb_mult = 2.0

    # WaveTrend Parameters
    n1 = 10
    n2 = 21
    obLevel1 = -53
    obLevel2 = -90

    # Calculate ATR (Wilder)
    tr1 = high - low
    tr2 = np.abs(high.shift(1) - close)
    tr3 = np.abs(low.shift(1) - close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/st_len, adjust=False).mean()

    # SuperTrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + st_mult * atr
    lower_band = hl2 - st_mult * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)

    prev_supertrend = None
    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = -1
        else:
            if close.iloc[i] > supertrend.iloc[i-1]:
                direction.iloc[i] = -1
            elif close.iloc[i] < supertrend.iloc[i-1]:
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = direction.iloc[i-1]

            if direction.iloc[i] == -1:
                supertrend.iloc[i] = lower_band.iloc[i]
                if upper_band.iloc[i] < supertrend.iloc[i-1] or direction.iloc[i-1] == 1:
                    supertrend.iloc[i] = supertrend.iloc[i-1]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                if lower_band.iloc[i] > supertrend.iloc[i-1] or direction.iloc[i-1] == -1:
                    supertrend.iloc[i] = supertrend.iloc[i-1]

    st_long = direction < 0
    st_short = direction > 0

    # ADX calculation
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    diff_high = high.diff()
    diff_low = -low.diff()

    plus_dm = pd.concat([diff_high, diff_low], axis=1).max(axis=1)
    plus_dm = plus_dm.where((diff_high > diff_low) & (diff_high > 0), 0.0)

    minus_dm = pd.concat([diff_low, diff_high], axis=1).max(axis=1)
    minus_dm = minus_dm.where((diff_low > diff_high) & (diff_low > 0), 0.0)

    di_plus = plus_dm.ewm(alpha=1.0/adx_len, adjust=False).mean()
    di_minus = minus_dm.ewm(alpha=1.0/adx_len, adjust=False).mean()

    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    adx = dx.ewm(alpha=1.0/adx_len, adjust=False).mean()
    strong_trend = adx > adx_thresh

    # Squeeze Momentum
    src = (high + low + close) / 3
    bb_ma = src.rolling(squeeze_len).mean()
    bb_sd = src.rolling(squeeze_len).std()
    bb_upper = bb_ma + bb_mult * bb_sd
    bb_lower = bb_ma - bb_mult * bb_sd

    kc_ma = src.rolling(squeeze_len).mean()
    kc_range = atr * kc_mult
    kc_upper = kc_ma + kc_range
    kc_lower = kc_ma - kc_range

    sqz_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    sqz_off = (bb_lower < kc_lower) | (bb_upper > kc_upper)

    sqz_on_prev = sqz_on.shift(1)
    sqz_release_long = sqz_off & sqz_on_prev
    sqz_release_short = sqz_off & sqz_on_prev

    # WaveTrend
    hlc3 = src
    esa = hlc3.ewm(span=n1, adjust=False).mean()
    d = np.abs(hlc3 - esa).ewm(span=n1, adjust=False).mean()
    ci = (esa - d) / (0.015 * d)
    tci = ci.ewm(span=n2, adjust=False).mean()

    wt_bull_cross = (tci > obLevel1) & (tci.shift(1) <= obLevel1)
    wt_bear_cross = (tci < obLevel1) & (tci.shift(1) >= obLevel1)

    # Entry conditions
    long_condition = st_long & strong_trend & sqz_release_long & wt_bull_cross
    short_condition = st_short & strong_trend & sqz_release_short & wt_bear_cross

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i < 2:
            continue
        if np.isnan(atr.iloc[i]) or np.isnan(adx.iloc[i]) or np.isnan(bb_ma.iloc[i]) or np.isnan(tci.iloc[i]):
            continue

        entry_price = close.iloc[i]
        entry_ts = int(time.iloc[i])

        if long_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        if short_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries