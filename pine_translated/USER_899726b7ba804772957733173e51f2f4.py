import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Ensure required columns exist
    if not all(col in df.columns for col in ['time','open','high','low','close','volume']):
        raise ValueError("DataFrame must contain columns: time, open, high, low, close, volume")
    
    # Work on a copy to avoid mutating input
    df = df.copy()
    
    # Convert time to datetime (UTC)
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['datetime'].dt.date
    
    # ------------------------------------------------------------------
    # Compute EMAs (Pine ta.ema(src, len) => src.ewm(span=len, adjust=False).mean())
    # ------------------------------------------------------------------
    df['fastEMA'] = df['close'].ewm(span=50, adjust=False).mean()
    df['slowEMA'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # ------------------------------------------------------------------
    # Compute previous day's high and low (daily high/low shifted by 1 day)
    # ------------------------------------------------------------------
    daily_high = df.groupby('date')['high'].max().rename('prevDayHigh')
    daily_low  = df.groupby('date')['low'].min().rename('prevDayLow')
    
    # Shift by one day to align previous day's values with current day
    daily_high_shifted = daily_high.shift(1)
    daily_low_shifted  = daily_low.shift(1)
    
    # Map back to original DataFrame
    df['prevDayHigh'] = df['date'].map(daily_high_shifted)
    df['prevDayLow']  = df['date'].map(daily_low_shifted)
    
    # ------------------------------------------------------------------
    # Determine trend: uptrend when fast EMA > slow EMA
    # ------------------------------------------------------------------
    df['uptrend'] = df['fastEMA'] > df['slowEMA']
    
    # ------------------------------------------------------------------
    # Detect sweeps of previous day's high/low
    # ------------------------------------------------------------------
    # Pine: close > prevDayHigh => flagpdh set; close < prevDayLow => flagpdl set
    df['sweep_high'] = df['close'] > df['prevDayHigh']
    df['sweep_low']  = df['close'] < df['prevDayLow']
    
    # ------------------------------------------------------------------
    # Build entry conditions
    # Long entry: sweep of previous day's low AND in an uptrend
    # Short entry: sweep of previous day's high AND NOT in an uptrend (downtrend)
    # ------------------------------------------------------------------
    df['long_cond']  = df['sweep_low'] & df['uptrend']
    df['short_cond'] = df['sweep_high'] & ~df['uptrend']
    
    # ------------------------------------------------------------------
    # Skip bars where required indicators are NaN (EMAs, prevDay values)
    # ------------------------------------------------------------------
    # We will filter rows where any of these are NaN for condition evaluation
    valid_rows = df.dropna(subset=['fastEMA','slowEMA','prevDayHigh','prevDayLow'])
    
    # ------------------------------------------------------------------
    # Find first entry per date for each direction
    # ------------------------------------------------------------------
    entries = []
    trade_num = 1
    
    # Process long entries
    long_df = valid_rows[valid_rows['long_cond']].copy()
    # Keep only the first occurrence per date
    long_first = long_df.groupby('date', sort=True).first().reset_index()
    for _, row in long_first.iterrows():
        ts = int(row['time'])
        entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = float(row['close'])
        entries.append({
            'trade_num': trade_num,
            'direction': 'long',
            'entry_ts': ts,
            'entry_time': entry_time_str,
            'entry_price_guess': entry_price,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': entry_price,
            'raw_price_b': entry_price
        })
        trade_num += 1
    
    # Process short entries
    short_df = valid_rows[valid_rows['short_cond']].copy()
    short_first = short_df.groupby('date', sort=True).first().reset_index()
    for _, row in short_first.iterrows():
        ts = int(row['time'])
        entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = float(row['close'])
        entries.append({
            'trade_num': trade_num,
            'direction': 'short',
            'entry_ts': ts,
            'entry_time': entry_time_str,
            'entry_price_guess': entry_price,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': entry_price,
            'raw_price_b': entry_price
        })
        trade_num += 1
    
    # Sort entries by entry_ts to maintain chronological order
    entries.sort(key=lambda x: x['entry_ts'])
    # Re-assign trade numbers after sorting
    for i, entry in enumerate(entries, start=1):
        entry['trade_num'] = i
    
    return entries