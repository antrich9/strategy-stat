import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time_dt', inplace=True)
    df.sort_index(inplace=True)
    
    htf = '240min'
    htf_ohlc = df[['open', 'high', 'low', 'close']].resample(htf).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    # Wilder ATR on HTF (period 144)
    tr1 = htf_ohlc['high'] - htf_ohlc['low']
    tr2 = abs(htf_ohlc['high'] - htf_ohlc['close'].shift(1))
    tr3 = abs(htf_ohlc['low'] - htf_ohlc['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    htf_atr = tr.ewm(alpha=1/144, adjust=False).mean()
    
    # Bullish FVG detection on HTF
    htf_ohlc['bull_condition'] = (htf_ohlc['low'] - htf_ohlc['high'].shift(2)) > htf_atr
    htf_ohlc['bullFvgUpper'] = np.where(htf_ohlc['bull_condition'], htf_ohlc['high'].shift(2), np.nan)
    htf_ohlc['bullFvgLower'] = np.where(htf_ohlc['bull_condition'], htf_ohlc['low'], np.nan)
    htf_ohlc['bullMidpoint'] = (htf_ohlc['bullFvgUpper'] + htf_ohlc['bullFvgLower']) / 2
    
    # Bearish FVG detection on HTF
    htf_ohlc['bear_condition'] = (htf_ohlc['low'].shift(2) - htf_ohlc['high']) > htf_atr
    htf_ohlc['bearFvgUpper'] = np.where(htf_ohlc['bear_condition'], htf_ohlc['high'], np.nan)
    htf_ohlc['bearFvgLower'] = np.where(htf_ohlc['bear_condition'], htf_ohlc['low'].shift(2), np.nan)
    htf_ohlc['bearMidpoint'] = (htf_ohlc['bearFvgUpper'] + htf_ohlc['bearFvgLower']) / 2
    
    # Forward fill FVG values to simulate ta.valuewhen behavior
    for col in ['bullMidpoint', 'bearMidpoint']:
        htf_ohlc[col] = htf_ohlc[col].ffill()
    
    # Merge HTF midpoints back to original timeframe using forward fill
    htf_aligned = htf_ohlc[['bullMidpoint', 'bearMidpoint']].reindex(df.index, method='ffill')
    df['bullMidpoint'] = htf_aligned['bullMidpoint']
    df['bearMidpoint'] = htf_aligned['bearMidpoint']
    
    # Build boolean series for entry conditions
    bull_cross = (df['low'] < df['bullMidpoint']) & (df['low'].shift(1) >= df['bullMidpoint'].shift(1))
    bear_cross = (df['high'] > df['bearMidpoint']) & (df['high'].shift(1) <= df['bearMidpoint'].shift(1))
    
    # Collect entry indices with direction
    bull_entries = [(i, 'long') for i in df.index[bull_cross] if pd.notna(df.loc[i, 'bullMidpoint'])]
    bear_entries = [(i, 'short') for i in df.index[bear_cross] if pd.notna(df.loc[i, 'bearMidpoint'])]
    
    entries = bull_entries + bear_entries
    entries.sort(key=lambda x: x[0])
    
    result = []
    trade_num = 1
    
    for idx, direction in entries:
        i = df.index.get_loc(idx)
        entry_ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        entry_price_guess = float(df['close'].iloc[i])
        
        raw_price_a = raw_price_b = entry_price_guess
        
        result.append({
            'trade_num': trade_num,
            'direction': direction,
            'entry_ts': entry_ts,
            'entry_time': entry_time,
            'entry_price_guess': entry_price_guess,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': raw_price_a,
            'raw_price_b': raw_price_b
        })
        trade_num += 1
    
    return result