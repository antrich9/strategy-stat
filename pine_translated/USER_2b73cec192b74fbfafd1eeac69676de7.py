import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    
    # Convert time to datetime (UTC to London timezone)
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert('Europe/London')
    
    # Calculate previous trading day high and low
    df['date'] = df['datetime'].dt.date
    daily = df.groupby('date').agg({
        'high': 'max',
        'low': 'min'
    }).reset_index()
    daily['prev_day_high'] = daily['high'].shift(1)
    daily['prev_day_low'] = daily['low'].shift(1)
    
    # Merge previous day values back to main df
    df = df.merge(daily[['date', 'prev_day_high', 'prev_day_low']], on='date', how='left')
    
    # Trading window: 07:45-09:45 and 15:45-16:45 London time
    hour = df['datetime'].dt.hour
    minute = df['datetime'].dt.minute
    time_in_minutes = hour * 60 + minute
    
    in_trading_window = (
        ((time_in_minutes >= 7*60 + 45) & (time_in_minutes <= 9*60 + 45)) |
        ((time_in_minutes >= 15*60 + 45) & (time_in_minutes <= 16*60 + 45))
    )
    
    # No Friday trading ("No FRIDY trading")
    is_friday = df['datetime'].dt.dayofweek == 4
    valid_day = ~is_friday
    
    # Entry signals using crossover/crossunder logic
    close = df['close']
    prev_close = close.shift(1)
    
    # Long: crossover previous day low
    # Short: crossunder previous day high
    long_condition = (close > df['prev_day_low']) & (prev_close <= df['prev_day_low'].shift(1))
    short_condition = (close < df['prev_day_high']) & (prev_close >= df['prev_day_high'].shift(1))
    
    # Apply filters
    long_signal = long_condition & in_trading_window & valid_day & df['prev_day_low'].notna()
    short_signal = short_condition & in_trading_window & valid_day & df['prev_day_high'].notna()
    
    # Generate entry list
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if long_signal.iloc[i]:
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
        elif short_signal.iloc[i]:
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