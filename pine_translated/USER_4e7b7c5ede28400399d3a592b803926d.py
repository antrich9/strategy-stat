import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('datetime', inplace=True)
    
    resampled = df.resample('240min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    tr1 = resampled['high'] - resampled['low']
    tr2 = abs(resampled['high'] - resampled['close'].shift(1))
    tr3 = abs(resampled['low'] - resampled['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_length = 20
    atr = pd.Series(index=resampled.index, dtype=float)
    atr.iloc[atr_length - 1] = tr.iloc[:atr_length].sum()
    for i in range(atr_length, len(tr)):
        atr.iloc[i] = (atr.iloc[i - 1] * (atr_length - 1) + tr.iloc[i]) / atr_length
    resampled['atr'] = atr
    
    resampled['vol_filter'] = resampled['volume'].shift(1) > resampled['volume'].rolling(9).mean() * 1.5
    resampled['atr_filter_val'] = resampled['atr'] / 1.5
    resampled['atr_filter'] = (
        (resampled['low'] - resampled['high'].shift(2) > resampled['atr_filter_val']) |
        (resampled['low'].shift(2) - resampled['high'] > resampled['atr_filter_val'])
    )
    resampled['trend_sma'] = resampled['close'].rolling(54).mean()
    resampled['trend_up'] = resampled['trend_sma'] > resampled['trend_sma'].shift(1)
    
    resampled['bullish_fvg'] = (
        (resampled['low'] > resampled['high'].shift(2)) &
        resampled['vol_filter'] &
        resampled['atr_filter'] &
        resampled['trend_up']
    )
    resampled['bearish_fvg'] = (
        (resampled['high'] < resampled['low'].shift(2)) &
        resampled['vol_filter'] &
        resampled['atr_filter'] &
        ~resampled['trend_up']
    )
    
    lastFVG = 0
    trade_num = 0
    entries = []
    
    for i in range(2, len(resampled)):
        bull_fvg = resampled['bullish_fvg'].iloc[i]
        bear_fvg = resampled['bearish_fvg'].iloc[i]
        
        if bull_fvg and lastFVG == -1:
            trade_num += 1
            ts = int(resampled.index[i].timestamp())
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(resampled['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(resampled['close'].iloc[i]),
                'raw_price_b': float(resampled['close'].iloc[i])
            })
            lastFVG = 1
        elif bear_fvg and lastFVG == 1:
            trade_num += 1
            ts = int(resampled.index[i].timestamp())
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(resampled['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(resampled['close'].iloc[i]),
                'raw_price_b': float(resampled['close'].iloc[i])
            })
            lastFVG = -1
        elif bull_fvg:
            lastFVG = 1
        elif bear_fvg:
            lastFVG = -1
    
    return entries