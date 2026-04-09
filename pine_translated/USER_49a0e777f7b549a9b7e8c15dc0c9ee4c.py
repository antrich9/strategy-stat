import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    fastEMA = df['close'].ewm(span=50, adjust=False).mean()
    slowEMA = df['close'].ewm(span=200, adjust=False).mean()
    df['crossUp'] = (fastEMA > slowEMA) & (fastEMA.shift(1) <= slowEMA.shift(1))
    df['crossDown'] = (fastEMA < slowEMA) & (fastEMA.shift(1) >= slowEMA.shift(1))
    daily = df.groupby('date').agg({'high': 'max', 'low': 'min'}).reset_index()
    daily['prevDayHigh'] = daily['high'].shift(1)
    daily['prevDayLow'] = daily['low'].shift(1)
    df = df.merge(daily[['date', 'prevDayHigh', 'prevDayLow']], on='date', how='left')
    df['pdh_swept'] = df['close'] > df['prevDayHigh']
    df['pdl_swept'] = df['close'] < df['prevDayLow']
    df['ob_bull'] = (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open']) & (df['close'] > df['high'].shift(1))
    df['ob_bear'] = (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open']) & (df['close'] < df['low'].shift(1))
    df['fvg_bull'] = df['low'] > df['high'].shift(2)
    df['fvg_bear'] = df['high'] < df['low'].shift(2)
    df['morning_session'] = (df['hour'] >= 7) & (df['hour'] < 10)
    df['afternoon_session'] = (df['hour'] >= 12) & (df['hour'] < 15)
    df['long_entry'] = df['crossUp'] & df['pdl_swept'] & df['morning_session']
    df['short_entry'] = df['crossDown'] & df['pdh_swept'] & df['afternoon_session']
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if pd.isna(df['fastEMA'].iloc[i]) or pd.isna(df['slowEMA'].iloc[i]) or pd.isna(df['prevDayHigh'].iloc[i]) or pd.isna(df['prevDayLow'].iloc[i]):
            continue
        if df['long_entry'].iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif df['short_entry'].iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    return entries