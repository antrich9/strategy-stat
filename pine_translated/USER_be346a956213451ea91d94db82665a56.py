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
    # Settings
    thresholdPer = 0.0
    auto = False
    
    # Prepare dataframe
    df = df.copy()
    df['ts_datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Get daily open for bias calculation
    df_indexed = df.set_index('ts_datetime')
    daily = df_indexed.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    daily = daily.dropna()
    
    daily_open_series = daily['open']
    df['daily_open'] = daily_open_series.reindex(df_indexed.index, method='ffill').values
    
    # Threshold calculation (auto vs manual)
    if auto:
        cumulative_val = ((df['high'] - df['low']) / df['low']).cumsum()
        bar_index_arr = np.arange(1, len(df) + 1)
        threshold = cumulative_val / bar_index_arr
    else:
        threshold = thresholdPer / 100.0
    
    # Calculate shifted values for FVG detection
    high_2 = df['high'].shift(2)
    low_2 = df['low'].shift(2)
    close_1 = df['close'].shift(1)
    
    # Bullish FVG: low > high[2] and close[1] > high[2] and (low - high[2]) / high[2] > threshold
    bull_fvg = (df['low'] > high_2) & (close_1 > high_2) & ((df['low'] - high_2) / high_2.replace(0, np.nan) > threshold)
    
    # Bearish FVG: high < low[2] and close[1] < low[2] and (low[2] - high) / low[2] > threshold
    bear_fvg = (df['high'] < low_2) & (close_1 < low_2) & ((low_2 - df['high']) / low_2.replace(0, np.nan) > threshold)
    
    # Daily bias conditions
    isDailyGreen = df['close'] > df['daily_open']
    isDailyRed = df['close'] < df['daily_open']
    
    bullishAllowed = isDailyRed
    bearishAllowed = isDailyGreen
    
    # Track consecutive FVG counts
    consecutiveBullCount = 0
    consecutiveBearCount = 0
    
    entries = []
    trade_num = 1
    
    for i in range(2, len(df)):
        if pd.isna(bull_fvg.iloc[i]) or pd.isna(bear_fvg.iloc[i]):
            continue
        
        bull_signal = bull_fvg.iloc[i]
        bear_signal = bear_fvg.iloc[i]
        
        if bull_signal:
            if bullishAllowed.iloc[i]:
                consecutiveBullCount += 1
                consecutiveBearCount = 0
                
                if consecutiveBullCount >= 2:
                    ts = int(df['time'].iloc[i])
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': ts,
                        'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                        'entry_price_guess': float(df['close'].iloc[i]),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(df['close'].iloc[i]),
                        'raw_price_b': float(df['close'].iloc[i])
                    })
                    trade_num += 1
                    consecutiveBullCount = 0
                    
        elif bear_signal:
            if bearishAllowed.iloc[i]:
                consecutiveBearCount += 1
                consecutiveBullCount = 0
                
                if consecutiveBearCount >= 2:
                    ts = int(df['time'].iloc[i])
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': ts,
                        'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                        'entry_price_guess': float(df['close'].iloc[i]),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(df['close'].iloc[i]),
                        'raw_price_b': float(df['close'].iloc[i])
                    })
                    trade_num += 1
                    consecutiveBearCount = 0
                    
        else:
            consecutiveBullCount = 0
            consecutiveBearCount = 0
    
    return entries