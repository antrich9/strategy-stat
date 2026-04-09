import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    is_downtrend = df['close'] < ema50
    support_level = df['close'].rolling(10).min()
    is_broken = df['close'] < support_level
    bearish_engulfing = (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open']) & (df['close'] < df['open'].shift(1))
    entry_condition = is_downtrend & is_broken & bearish_engulfing
    
    trades = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(support_level.iloc[i]):
            continue
        if entry_condition.iloc[i]:
            ts = df['time'].iloc[i]
            trades.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return trades