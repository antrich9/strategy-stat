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
    
    # Ensure datetime type
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('datetime', inplace=True)
    
    # Resample to 1-hour for EMA calculations (matches Pine Script request.security 60)
    hourly = df.resample('h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    hourly = hourly.dropna(subset=['close'])
    
    # Calculate EMAs on 1-hour close
    ema9 = hourly['close'].ewm(span=9, adjust=False).mean()
    ema18 = hourly['close'].ewm(span=18, adjust=False).mean()
    
    # Wilder RSI implementation
    def wilder_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi14 = wilder_rsi(hourly['close'], 14)
    
    # Merge hourly indicators back to original df index
    df['ema9'] = ema9
    df['ema18'] = ema18
    df['rsi14'] = rsi14
    df['ema9'] = df['ema9'].ffill()
    df['ema18'] = df['ema18'].ffill()
    df['rsi14'] = df['rsi14'].ffill()
    
    # Time window check in Europe/London timezone
    df['datetime_london'] = df.index.tz_convert('Europe/London')
    df['hour'] = df['datetime_london'].hour
    df['minute'] = df['datetime_london'].minute
    df['total_minutes'] = df['hour'] * 60 + df['minute']
    
    # Morning window: 07:45 to 09:45 (inclusive start, exclusive end)
    morning_start = 7 * 60 + 45  # 465
    morning_end = 9 * 60 + 45    # 585
    is_morning = (df['total_minutes'] >= morning_start) & (df['total_minutes'] < morning_end)
    
    # Afternoon window: 15:45 to 16:45 (inclusive start, exclusive end)
    afternoon_start = 15 * 60 + 45  # 945
    afternoon_end = 16 * 60 + 45    # 1005
    is_afternoon = (df['total_minutes'] >= afternoon_start) & (df['total_minutes'] < afternoon_end)
    
    df['in_time_window'] = is_morning | is_afternoon
    
    # Entry conditions
    df['condition_long'] = (df['close'] > df['ema9']) & (df['close'] > df['ema18'])
    df['condition_short'] = (df['close'] < df['ema9']) & (df['close'] < df['ema18'])
    
    df['long_signal'] = df['condition_long'] & df['in_time_window']
    df['short_signal'] = df['condition_short'] & df['in_time_window']
    
    # Filter out rows with NaN indicators
    valid_mask = df['ema9'].notna() & df['ema18'].notna() & df['rsi14'].notna()
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if not valid_mask.iloc[i]:
            continue
        
        ts = int(df['time'].iloc[i])
        price = float(df['close'].iloc[i])
        
        if df['long_signal'].iloc[i]:
            direction = 'long'
            entry = {
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            }
            entries.append(entry)
            trade_num += 1
        elif df['short_signal'].iloc[i]:
            direction = 'short'
            entry = {
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            }
            entries.append(entry)
            trade_num += 1
    
    return entries