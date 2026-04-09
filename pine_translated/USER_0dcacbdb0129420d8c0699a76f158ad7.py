import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']

    ema50 = close.ewm(span=50, adjust=False).mean()

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atrValue = pd.Series(tr).ewm(span=14, adjust=False).mean()

    window = 14
    window_minus_1 = window - 1

    highest_high = high.rolling(window=window, min_periods=window).max()
    aroonUp_pos = high.rolling(window=window, min_periods=window).apply(
        lambda x: (window_minus_1 - (np.argmax(x) if not np.isnan(x).all() else window_minus_1)) / window if not np.isnan(x).all() else np.nan,
        raw=False
    )
    aroonUp = (1 - aroonUp_pos) * 100

    lowest_low = low.rolling(window=window, min_periods=window).min()
    aroonDown_pos = low.rolling(window=window, min_periods=window).apply(
        lambda x: (window_minus_1 - (np.argmin(x) if not np.isnan(x).all() else window_minus_1)) / window if not np.isnan(x).all() else np.nan,
        raw=False
    )
    aroonDown = (1 - aroonDown_pos) * 100

    isHL = low > low.shift(1).rolling(2).min()

    bullishAroon = aroonUp > aroonDown
    bullishEMA = close > ema50

    longCondition = bullishEMA & bullishAroon & isHL

    results = []
    trade_num = 1
    for i in range(len(df)):
        if longCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price_guess = float(close.iloc[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1
    return results