import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    trades = []
    trade_num = 0

    lengthDaily = 50
    length4H = 200

    emaDaily = df['close'].ewm(span=lengthDaily, adjust=False).mean()
    ema4H = df['close'].ewm(span=length4H, adjust=False).mean()

    bullish = (df['close'] > emaDaily) & (df['close'] > ema4H)
    bearish = (df['close'] < emaDaily) & (df['close'] < ema4H)

    in_position = False
    for i in range(1, len(df)):
        if bullish.iloc[i] and not in_position:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            trades.append({
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
            in_position = True
        elif bearish.iloc[i] and not in_position:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            trades.append({
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
            in_position = True

    return trades