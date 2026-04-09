import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['datetime'].dt.date
    
    daily_agg = df.groupby('date').agg({
        'high': 'max',
        'low': 'min'
    }).reset_index()
    daily_agg.columns = ['date', 'daily_high', 'daily_low']
    daily_agg['prev_day_high'] = daily_agg['daily_high'].shift(1)
    daily_agg['prev_day_low'] = daily_agg['daily_low'].shift(1)
    df = df.merge(daily_agg[['date', 'prev_day_high', 'prev_day_low']], on='date', how='left')
    
    df['swing_high'] = df['high'].rolling(5).max()
    df['swing_low'] = df['low'].rolling(5).min()
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    df['atr'] = atr
    df['atr50'] = atr.rolling(50).mean()
    df['validVolatility'] = df['atr'] > 0.5 * df['atr50']
    
    df['sweptHigh'] = df['swing_high'] > df['prev_day_high']
    df['sweptLow'] = df['swing_low'] < df['prev_day_low']
    
    crossover_cond = (close > df['prev_day_low']) & (close.shift(1) <= df['prev_day_low'].shift(1))
    crossunder_cond = (close < df['prev_day_high']) & (close.shift(1) >= df['prev_day_high'].shift(1))
    
    longCondition = df['sweptLow'] & crossover_cond & df['validVolatility']
    shortCondition = df['sweptHigh'] & crossunder_cond & df['validVolatility']
    
    entries = []
    trade_num = 1
    
    for i in range(5, len(df)):
        if pd.isna(df['atr'].iloc[i]) or pd.isna(df['atr50'].iloc[i]):
            continue
        if longCondition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': df['datetime'].iloc[i].isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif shortCondition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': df['datetime'].iloc[i].isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries