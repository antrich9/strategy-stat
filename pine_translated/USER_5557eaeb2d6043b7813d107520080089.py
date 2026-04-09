import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    length_t3 = 5
    factor_t3 = 0.7

    def gd_t3(src, length, factor):
        ema = src.ewm(span=length, adjust=False).mean()
        ema_of_ema = ema.ewm(span=length, adjust=False).mean()
        return ema * (1 + factor) - ema_of_ema * factor

    src_close = df['close']
    t3 = gd_t3(gd_t3(gd_t3(src_close, length_t3, factor_t3), length_t3, factor_t3), length_t3, factor_t3)

    t3_prev = t3.shift(1)
    t3_signals = pd.Series(np.where(t3 > t3_prev, 1, -1), index=t3.index)

    valid_t3 = t3.notna()
    basic_long = valid_t3 & (t3_signals > 0) & (src_close > t3)
    basic_short = valid_t3 & (t3_signals < 0) & (src_close < t3)

    entries = []
    trade_num = 1
    for i in range(len(df)):
        if basic_long.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif basic_short.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
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