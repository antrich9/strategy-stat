import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    high = df['high']
    low = df['low']
    close = df['close']

    # Bullish Fair Value Gap (FVG): low > high[2] and close[1] > high[2]
    bull_fvg = (low > high.shift(2)) & (close.shift(1) > high.shift(2))
    # Bearish Fair Value Gap (FVG): high < low[2] and close[1] < low[2]
    bear_fvg = (high < low.shift(2)) & (close.shift(1) < low.shift(2))

    # Fill NaN results with False so they are ignored
    bull_fvg = bull_fvg.fillna(False)
    bear_fvg = bear_fvg.fillna(False)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if bull_fvg.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
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

        if bear_fvg.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = float(df['close'].iloc[i])
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