import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Strategy parameters
    fast_len = 8
    med_len = 20
    slow_len = 50
    doji_perc = 0.30

    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']

    # EMAs
    ema8 = close.ewm(span=fast_len, adjust=False).mean()
    ema20 = close.ewm(span=med_len, adjust=False).mean()
    ema50 = close.ewm(span=slow_len, adjust=False).mean()

    # Doji detection
    body = (close - open_).abs()
    rng = high - low
    ratio = body / rng
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    is_doji = (rng > 0) & (ratio <= doji_perc)

    # EMA trend
    ema_bull = (ema8 > ema20) & (ema20 > ema50)
    ema_bear = (ema8 < ema20) & (ema20 < ema50)

    # Price location within range
    close_in_upper33 = ((close - low) / rng) >= 0.67
    close_in_lower33 = ((high - close) / rng) >= 0.67

    body_in_upper33 = (open_ >= low + rng * 0.67) & (close >= low + rng * 0.67)
    body_in_lower33 = (open_ <= low + rng * 0.33) & (close <= low + rng * 0.33)

    # Signal conditions
    bull_signal = ema_bull & is_doji & (low <= ema8) & close_in_upper33 & body_in_upper33
    bear_signal = ema_bear & is_doji & (high >= ema8) & close_in_lower33 & body_in_lower33

    bull_only = bull_signal & ~bear_signal
    bear_only = bear_signal & ~bull_signal

    entries = []
    trade_num = 1

    for i in df.index:
        if bull_only.loc[i]:
            ts = int(df.loc[i, 'time'])
            entry_price = float(df.loc[i, 'close'])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif bear_only.loc[i]:
            ts = int(df.loc[i, 'time'])
            entry_price = float(df.loc[i, 'close'])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries