import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Parameters
    PP = 7  # Pivot period for order blocks detector
    atr_period = 14  # ATR period for stop loss
    atr_multiplier = 1.5  # ATR multiplier for stop loss distance (not used for entry but for reference)
    take_profit_multiplier = 2  # 2R take profit (not used for entry)
    
    # Ensure required columns exist
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("DataFrame must contain columns: time, open, high, low, close, volume")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Compute pivot high and low using rolling window
    # A bar is a pivot high if it is the highest in the window of period*2+1
    period = PP
    window = period * 2 + 1
    
    # Compute rolling max and min
    df['roll_max'] = df['high'].rolling(window=window, center=True).max()
    df['roll_min'] = df['low'].rolling(window=window, center=True).min()
    
    # Determine pivot high/low boolean
    df['pivot_high'] = (df['high'] == df['roll_max']) & (df['roll_max'].notna())
    df['pivot_low'] = (df['low'] == df['roll_min']) & (df['roll_min'].notna())
    
    # Initialize major high/low series with NaN
    df['major_high'] = np.nan
    df['major_low'] = np.nan
    
    # We will forward fill the most recent pivot high/low
    # But we also want to store only the most recent pivot high/low that is not NaN
    # We can use ffill for pivot high and low separately
    df['pivot_high_val'] = np.where(df['pivot_high'], df['high'], np.nan)
    df['pivot_low_val'] = np.where(df['pivot_low'], df['low'], np.nan)
    
    # Forward fill to get the most recent pivot high/low
    df['major_high'] = df['pivot_high_val'].ffill()
    df['major_low'] = df['pivot_low_val'].ffill()
    
    # Compute ATR using Wilder's method
    def wilder_atr(dataframe, period):
        high = dataframe['high']
        low = dataframe['low']
        close = dataframe['close']
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    df['atr'] = wilder_atr(df, atr_period)
    
    # Compute entry conditions
    # Long entry: price crosses above major low (crossover)
    # Short entry: price crosses below major high (crossunder)
    # However, we also want to ensure that major low/high are defined (not NaN) and ATR is defined
    # We also want to avoid generating entries on the same bar where the pivot is identified (maybe we need to shift)
    
    # For crossover: close > major_low and previous close <= major_low
    # For crossunder: close < major_high and previous close >= major_high
    df['prev_close'] = df['close'].shift(1)
    df['prev_major_low'] = df['major_low'].shift(1)
    df['prev_major_high'] = df['major_high'].shift(1)
    
    # Conditions
    df['long_condition'] = (df['close'] > df['major_low']) & (df['prev_close'] <= df['prev_major_low'])
    df['short_condition'] = (df['close'] < df['major_high']) & (df['prev_close'] >= df['prev_major_high'])
    
    # We might also want to ensure that the major low/high are not NaN at the current and previous bar
    df['long_condition'] = df['long_condition'] & df['major_low'].notna() & df['prev_major_low'].notna()
    df['short_condition'] = df['short_condition'] & df['major_high'].notna() & df['prev_major_high'].notna()
    
    # We might also want to ensure ATR is not NaN
    df['long_condition'] = df['long_condition'] & df['atr'].notna()
    df['short_condition'] = df['short_condition'] & df['atr'].notna()
    
    # Initialize list to store entries
    entries = []
    trade_num = 1
    
    # Iterate over rows
    for i in range(len(df)):
        # Check for long entry
        if df['long_condition'].iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        # Check for short entry
        elif df['short_condition'].iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries