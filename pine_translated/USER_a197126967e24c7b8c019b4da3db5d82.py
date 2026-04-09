import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    Periods = 10
    Multiplier = 3.0
    
    ema8 = df['close'].ewm(span=8, adjust=False).mean()
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    
    high = df['high']
    low = df['low']
    close = df['close']
    hl2 = (high + low) / 2
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = pd.Series(index=df.index, dtype=float)
    atr.iloc[Periods-1] = tr.iloc[:Periods].mean()
    alpha = 1.0 / Periods
    for i in range(Periods, len(df)):
        atr.iloc[i] = tr.iloc[i] * alpha + atr.iloc[i-1] * (1 - alpha)
    
    up = pd.Series(index=df.index, dtype=float)
    dn = pd.Series(index=df.index, dtype=float)
    up[Periods-1] = hl2.iloc[Periods-1] - Multiplier * atr.iloc[Periods-1]
    dn[Periods-1] = hl2.iloc[Periods-1] + Multiplier * atr.iloc[Periods-1]
    
    for i in range(Periods, len(df)):
        up_val = hl2.iloc[i] - Multiplier * atr.iloc[i]
        if not pd.isna(up.iloc[i-1]):
            up_prev = up.iloc[i-1]
            if close.iloc[i-1] > up_prev:
                up_val = max(up_val, up_prev)
        up.iloc[i] = up_val
        
        dn_val = hl2.iloc[i] + Multiplier * atr.iloc[i]
        if not pd.isna(dn.iloc[i-1]):
            dn_prev = dn.iloc[i-1]
            if close.iloc[i-1] < dn_prev:
                dn_val = min(dn_val, dn_prev)
        dn.iloc[i] = dn_val
    
    trend = pd.Series(0, index=df.index, dtype=int)
    trend.iloc[Periods-1] = 1
    
    for i in range(Periods, len(df)):
        if pd.isna(trend.iloc[i-1]) or pd.isna(close.iloc[i-1]) or pd.isna(close.iloc[i-2]):
            trend.iloc[i] = 0
        elif trend.iloc[i-1] == -1 and close.iloc[i-1] > dn.iloc[i-2]:
            trend.iloc[i] = 1
        elif trend.iloc[i-1] == 1 and close.iloc[i-1] < up.iloc[i-2]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
    
    buy_signal = pd.Series(False, index=df.index)
    sell_signal = pd.Series(False, index=df.index)
    
    for i in range(Periods + 1, len(df)):
        buy_signal.iloc[i] = trend.iloc[i] == 1 and trend.iloc[i-1] == -1
        sell_signal.iloc[i] = trend.iloc[i] == -1 and trend.iloc[i-1] == 1
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(buy_signal.iloc[i]) or pd.isna(sell_signal.iloc[i]):
            continue
        if buy_signal.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif sell_signal.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries