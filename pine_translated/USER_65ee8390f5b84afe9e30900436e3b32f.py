import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    open_price = df['open']
    high = df['high']
    low = df['low']
    
    # Calculate EMAs
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    
    # Pin bar parameters
    pin_bar_size = 30.0  # default input
    
    # Calculate body and wick sizes
    body_size = (close - open_price).abs()
    
    lower_wick_size = np.where(open_price > close, close - low, open_price - low)
    upper_wick_size = np.where(open_price > close, high - open_price, high - close)
    
    # Bullish Pin Bar: close > open, lower wick > body * pin_bar_size/100, lower wick > 2 * upper wick
    is_bullish_pin_bar = (close > open_price) & \
                         (lower_wick_size > body_size * pin_bar_size / 100.0) & \
                         (lower_wick_size > upper_wick_size * 2.0)
    
    # Bearish Pin Bar: close < open, upper wick > body * pin_bar_size/100, upper wick > 2 * lower wick
    is_bearish_pin_bar = (close < open_price) & \
                         (upper_wick_size > body_size * pin_bar_size / 100.0) & \
                         (upper_wick_size > lower_wick_size * 2.0)
    
    # Long entry conditions: Bullish pin bar with close[1] > ema8 and EMAs in ascending order
    long_conditions = is_bullish_pin_bar & \
                       (close.shift(1) > ema8.shift(1)) & \
                       (ema8 > ema20) & \
                       (ema20 > ema50)
    
    # Short entry conditions: Bearish pin bar with close[1] < ema8 and EMAs in descending order
    short_conditions = is_bearish_pin_bar & \
                       (close.shift(1) < ema8.shift(1)) & \
                       (ema8 < ema20) & \
                       (ema20 < ema50)
    
    # Estimate mintick as minimum tick (0.01 is a reasonable default for many instruments)
    mintick = 0.01
    
    entries = []
    trade_num = 1
    
    # Start from index 50 to skip bars where ema50 might still be NaN
    for i in range(50, len(df)):
        if pd.isna(ema50.iloc[i]):
            continue
        
        if long_conditions.iloc[i]:
            entry_price = float(high.iloc[i] + mintick * 2)
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_conditions.iloc[i]:
            entry_price = float(low.iloc[i] - mintick * 2)
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries