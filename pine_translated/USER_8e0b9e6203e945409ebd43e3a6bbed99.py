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
    # ---------- helper: True Range ----------
    def true_range(hi, lo, cl):
        prev_cl = cl.shift(1).fillna(cl)
        tr1 = hi - lo
        tr2 = (hi - prev_cl).abs()
        tr3 = (lo - prev_cl).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ---------- helper: Wilder ATR ----------
    def wilder_atr(tr, period):
        # Wilder smoothing via ewm with alpha = 1/period
        return tr.ewm(alpha=1.0 / period, adjust=False).mean()

    # ---------- helper: Supertrend ----------
    def supertrend(hi, lo, cl, atr_series, period, mult):
        hl2 = (hi + lo) * 0.5
        upperband = hl2 + mult * atr_series
        lowerband = hl2 - mult * atr_series

        direction = pd.Series(1, index=cl.index)      # 1 = bullish, -1 = bearish
        supertrend_val = pd.Series(np.nan, index=cl.index)

        # initial values
        supertrend_val.iloc[0] = lowerband.iloc[0]
        direction.iloc[0] = 1

        up = lowerband.copy()
        dn = upperband.copy()

        for i in range(1, len(cl)):
            if pd.isna(atr_series.iloc[i]) or pd.isna(atr_series.iloc[i - 1]):
                # keep previous state
                direction.iloc[i] = direction.iloc[i - 1]
                supertrend_val.iloc[i] = supertrend_val.iloc[i - 1]
                up.iloc[i] = up.iloc[i - 1]
                dn.iloc[i] = dn.iloc[i - 1]
                continue

            # update up/down bands
            if direction.iloc[i - 1] == 1:
                up.iloc[i] = max(lowerband.iloc[i], up.iloc[i - 1])
            else:
                up.iloc[i] = lowerband.iloc[i]

            if direction.iloc[i - 1] == -1:
                dn.iloc[i] = min(upperband.iloc[i], dn.iloc[i - 1])
            else:
                dn.iloc[i] = upperband.iloc[i]

            # reversal checks
            if direction.iloc[i - 1] == 1 and cl.iloc[i] < dn.iloc[i]:
                direction.iloc[i] = -1
                supertrend_val.iloc[i] = up.iloc[i]
            elif direction.iloc[i - 1] == -1 and cl.iloc[i] > up.iloc[i]:
                direction.iloc[i] = 1
                supertrend_val.iloc[i] = dn.iloc[i]
            else:
                direction.iloc[i] = direction.iloc[i - 1]
                supertrend_val.iloc[i] = supertrend_val.iloc[i - 1]

        return direction

    # ---------- indicator parameters ----------
    ATR_PERIOD = 14
    EMA_LENGTH = 200
    ST_PERIOD = 7
    ST_MULTIPLIER = 1

    # ---------- compute indicators ----------
    tr = true_range(df['high'], df['low'], df['close'])
    atr = wilder_atr(tr, ATR_PERIOD)

    ema = df['close'].ewm(span=EMA_LENGTH, adjust=False).mean()

    st_dir = supertrend(df['high'], df['low'], df['close'], atr, ST_PERIOD, ST_MULTIPLIER)

    # ---------- entry conditions ----------
    is_ema_bullish = df['close'] > ema
    is_ema_bearish = df['close'] < ema

    is_st_bullish = st_dir == 1
    is_st_bearish = st_dir == -1

    long_cond  = is_ema_bullish & is_st_bullish
    short_cond = is_ema_bearish & is_st_bearish

    # ---------- generate entries ----------
    entries = []
    trade_num = 1

    for i in range(len(df)):
        # skip bars where needed indicators are NaN
        if (pd.isna(atr.iloc[i]) or pd.isna(ema.iloc[i]) or pd.isna(st_dir.iloc[i])):
            continue

        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries