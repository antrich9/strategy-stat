import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Constants
    pip_size = 0.0002  # Assume non-JPY pair; adjust if needed
    start_ts = int(datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime(2023, 12, 31, 23, 59, 0, tzinfo=timezone.utc).timestamp())
    
    # Indicators
    fastEMA = df['close'].ewm(span=8, adjust=False).mean()
    mediumEMA = df['close'].ewm(span=20, adjust=False).mean()
    slowEMA = df['close'].ewm(span=50, adjust=False).mean()
    
    # Wilder RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_value = 100 - (100 / (1 + rs))
    
    # Thresholds
    upper_threshold = df['high'] - (df['high'] - df['low']) * 0.31
    lower_threshold = df['low'] + (df['high'] - df['low']) * 0.31
    
    # Candle conditions
    bullish_candle = (df['close'] > upper_threshold) & (df['open'] > upper_threshold) & (df['low'] <= fastEMA)
    bearish_candle = (df['close'] < lower_threshold) & (df['open'] < lower_threshold) & (df['high'] >= fastEMA)
    
    # EMA alignment
    long_EMAs_aligned = (fastEMA > mediumEMA) & (mediumEMA > slowEMA)
    short_EMAs_aligned = (fastEMA < mediumEMA) & (mediumEMA < slowEMA)
    
    # Session check (London 7:00-16:00)
    ts = pd.to_datetime(df['time'], unit='s')
    in_london = (ts.dt.hour >= 7) & ((ts.dt.hour < 16) | ((ts.dt.hour == 16) & (ts.dt.minute == 0)))
    
    # Date range
    in_date_range = (df['time'] >= start_ts) & (df['time'] <= end_ts)
    
    # Entry conditions
    long_condition = bullish_candle & long_EMAs_aligned & (rsi_value < 70) & in_london & in_date_range
    short_condition = bearish_candle & short_EMAs_aligned & (rsi_value > 30) & in_london & in_date_range
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        # Skip NaN bars for required indicators
        if pd.isna(fastEMA.iloc[i]) or pd.isna(mediumEMA.iloc[i]) or pd.isna(slowEMA.iloc[i]) or pd.isna(rsi_value.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            entry_price = df['high'].iloc[i] + pip_size
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
        
        if short_condition.iloc[i]:
            entry_price = df['low'].iloc[i] - pip_size
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