import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Strategy parameters
    PDCM = 0.7          # Min Body % (70%)
    CDBA = 100          # ATR Length
    FDB = 1.3           # Elephant Factor

    # Price series
    open_ = df['open']
    high = df['high']
    low = df['low']
    close = df['close']

    # Fair Value Gap (FVG) detection
    bull_fvg = low > high.shift(2)
    bear_fvg = high < low.shift(2)

    # True Range for ATR
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - close.shift(1)),
            np.abs(low - close.shift(1))
        )
    )

    # Wilder ATR (alpha = 1 / length)
    atr = tr.ewm(alpha=1.0 / CDBA, adjust=False).mean()
    atr_prev = atr.shift(1)

    # Body size and its percentage of the candle range
    body = close - open_
    body_abs = body.abs()
    range_size = high - low
    body_pct = np.where(range_size > 0, body_abs / range_size, 0.0)

    # Elephant candle conditions
    is_bull_elephant = (
        (body > 0) &
        (body_pct >= PDCM) &
        (body_abs >= atr_prev * FDB)
    )
    is_bear_elephant = (
        (body < 0) &
        (body_pct >= PDCM) &
        (body_abs >= atr_prev * FDB)
    )

    # Entry signals: FVG must have been present on the previous bar
    long_condition = bull_fvg.shift(1) & is_bull_elephant
    short_condition = bear_fvg.shift(1) & is_bear_elephant

    # Ensure NaN values are treated as False
    long_condition = long_condition.fillna(False)
    short_condition = short_condition.fillna(False)

    # Build entry list
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if long_condition.iloc[i]:
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
        if short_condition.iloc[i]:
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