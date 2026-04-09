import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.

    Returns list of dicts:
    [{'trade_num': int, 'direction': 'long' or 'short',
      'entry_ts': int, 'entry_time': str,
      'entry_price_guess': float,
      'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
      'raw_price_a': float, 'raw_price_b': float}]
    """
    
    # Helper function to get previous day's start/end timestamps
    def get_prev_day_bounds(timestamps):
        day_bounds = []
        current_day = None
        for ts in timestamps:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start.replace(hour=23, minute=59, second=59, microsecond=999999)
            if current_day != day_start:
                current_day = day_start
            day_bounds.append((day_start.timestamp(), day_end.timestamp()))
        return day_bounds
    
    results = []
    trade_num = 1
    
    # Get previous day high and low for each bar
    prev_day_high = np.zeros(len(df))
    prev_day_low = np.zeros(len(df))
    
    timestamps = df['time'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    day_bounds = get_prev_day_bounds(timestamps)
    
    for i in range(1, len(df)):
        pdh = -np.inf
        pdl = np.inf
        current_ts = timestamps[i]
        
        for j in range(i):
            if j < i and day_bounds[j][0] < current_ts:
                if highs[j] > pdh:
                    pdh = highs[j]
                if lows[j] < pdl:
                    pdl = lows[j]
        
        prev_day_high[i] = pdh if pdh != -np.inf else np.nan
        prev_day_low[i] = pdl if pdl != np.inf else np.nan
    
    prev_day_high_series = pd.Series(prev_day_high)
    prev_day_low_series = pd.Series(prev_day_low)
    close_series = pd.Series(closes)
    high_series = pd.Series(highs)
    low_series = pd.Series(lows)
    
    # Detect PDH and PDL raids (using previous day values)
    pdh_hit = high_series >= prev_day_high_series
    pdl_hit = low_series <= prev_day_low_series
    
    # Determine bias: bullish (1) if close > prev_day_high, bearish (-1) if close < prev_day_low
    bullish_bias = close_series > prev_day_high_series
    bearish_bias = close_series < prev_day_low_series
    
    # Detect new day (when day changes)
    new_day = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        dt_current = datetime.fromtimestamp(timestamps[i], tz=timezone.utc)
        dt_prev = datetime.fromtimestamp(timestamps[i-1], tz=timezone.utc)
        if dt_current.date() != dt_prev.date():
            new_day[i] = True
    
    new_day_series = pd.Series(new_day)
    
    # Track bias state
    daily_bias = np.zeros(len(df))
    prev_bias = 0
    
    for i in range(1, len(df)):
        if new_day_series.iloc[i]:
            prev_bias = 0
        
        if bullish_bias.iloc[i] and not pd.isna(prev_day_high_series.iloc[i]):
            daily_bias[i] = 1
        elif bearish_bias.iloc[i] and not pd.isna(prev_day_low_series.iloc[i]):
            daily_bias[i] = -1
        else:
            daily_bias[i] = prev_bias
        
        if daily_bias[i] != 0:
            prev_bias = daily_bias[i]
    
    daily_bias_series = pd.Series(daily_bias)
    
    # Build entry conditions
    # Short entry: PDH raid (high hits prev_day_high) with bearish bias
    # Long entry: PDL raid (low hits prev_day_low) with bullish bias
    short_entry = pdh_hit & (daily_bias_series == -1)
    long_entry = pdl_hit & (daily_bias_series == 1)
    
    # Combine conditions with valid data check
    valid_pdh = ~pd.isna(prev_day_high_series) & (prev_day_high_series != -np.inf)
    valid_pdl = ~pd.isna(prev_day_low_series) & (prev_day_low_series != np.inf)
    
    short_entry = short_entry & valid_pdh
    long_entry = long_entry & valid_pdl
    
    # Iterate and generate entries
    for i in range(1, len(df)):
        if pd.isna(high_series.iloc[i]) or pd.isna(low_series.iloc[i]) or pd.isna(closes.iloc[i]):
            continue
        
        entry_ts = int(timestamps[i])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        
        if short_entry.iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(closes[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(closes[i]),
                'raw_price_b': float(closes[i])
            })
            trade_num += 1
        
        if long_entry.iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(closes[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(closes[i]),
                'raw_price_b': float(closes[i])
            })
            trade_num += 1
    
    return results