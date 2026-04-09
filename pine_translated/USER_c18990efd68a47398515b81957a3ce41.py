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
    # ── Indicator parameters (mirroring Pine Script inputs) ──────────────────
    sensitivity   = 150
    fastLength    = 20
    slowLength    = 40
    channelLength = 20
    mult          = 2.0

    close = df['close']

    # ── MACD (EMA‑based) ───────────────────────────────────────────────────────
    fastEMA = close.ewm(span=fastLength, adjust=False).mean()
    slowEMA = close.ewm(span=slowLength, adjust=False).mean()
    macd    = fastEMA - slowEMA

    # t1 = (MACD today – MACD yesterday) * sensitivity
    t1 = (macd - macd.shift(1)) * sensitivity

    # ── Bollinger Band width (BBUpper – BBLower) ───────────────────────────────
    # BBUpper = basis + dev, BBLower = basis – dev  →  width = 2·dev
    roll_std = close.rolling(window=channelLength).std(ddof=1)   # sample stdev
    e1 = 2.0 * mult * roll_std

    # ── Trend indicator (only positive values) ───────────────────────────────
    trendUp = t1.where(t1 >= 0, 0.0)

    # ── Entry logic ───────────────────────────────────────────────────────────
    entry_made  = False
    trade_num   = 1
    entries     = []

    for i in range(1, len(df)):
        tr_up_now  = trendUp.iloc[i]
        e1_now     = e1.iloc[i]
        tr_up_prev = trendUp.iloc[i - 1]
        e1_prev    = e1.iloc[i - 1]

        # Skip bars with insufficient data (NaN)
        if (pd.isna(tr_up_now) or pd.isna(e1_now) or
            pd.isna(tr_up_prev) or pd.isna(e1_prev)):
            continue

        # Reset flag when trend drops below explosion line
        if tr_up_now < e1_now:
            entry_made = False

        # Long entry condition
        if (not entry_made) and (tr_up_now > e1_now) and (tr_up_prev <= e1_prev):
            entry_ts       = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price    = float(close.iloc[i])

            entries.append({
                'trade_num':          trade_num,
                'direction':          'long',
                'entry_ts':           entry_ts,
                'entry_time':         entry_time_str,
                'entry_price_guess':  entry_price,
                'exit_ts':            0,
                'exit_time':          '',
                'exit_price_guess':   0.0,
                'raw_price_a':        entry_price,
                'raw_price_b':        entry_price
            })

            trade_num  += 1
            entry_made = True

    return entries