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
    
    # Get previous day's high and low
    # In Pine: request.security(syminfo.tickerid, 'D', high[1], lookahead=barmerge.lookahead_on)
    # We need to resample to daily timeframe to get previous day high/low
    
    # Convert time to datetime
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['datetime'].dt.date
    
    # Calculate daily high and low for each day
    daily = df.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'open': 'first',
        'close': 'last'
    }).reset_index()
    
    # Get previous day's high and low
    daily['prev_day_high'] = daily['high'].shift(1)
    daily['prev_day_low'] = daily['low'].shift(1)
    daily['prev_day_close'] = daily['close'].shift(1)
    
    # Merge back to original dataframe
    df = df.merge(daily[['date', 'prev_day_high', 'prev_day_low', 'prev_day_close']], on='date', how='left')
    
    # Detect new day
    df['new_day'] = df['date'] != df['date'].shift(1)
    
    # Calculate bias based on close relative to previous day high/low
    # Bullish bias: close > prev_day_high
    # Bearish bias: close < prev_day_low
    
    df['above_pdh'] = df['close'] > df['prev_day_high']
    df['below_pdl'] = df['close'] < df['prev_day_low']
    
    # Detect hit on previous day high/low
    df['hit_pdh'] = df['high'] >= df['prev_day_high']
    df['hit_pdl'] = df['low'] <= df['prev_day_low']
    
    # Calculate daily bias
    df['bias'] = 0
    df.loc[df['close'] > df['prev_day_high'], 'bias'] = 1
    df.loc[df['close'] < df['prev_day_low'], 'bias'] = -1
    
    # Detect bias changes (for entries)
    df['prev_bias'] = df['bias'].shift(1)
    df['bias_change'] = (df['bias'] != df['prev_bias']) & (df['bias'] != 0)
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if df.iloc[i]['bias_change']:
            if df.iloc[i]['bias'] == 1:
                # Bullish bias - could be long entry
                # But title says "trend sell entry" when pdh == 1
                # So maybe short entry when bias becomes 1?
                # Or maybe the reverse: buy when bias becomes -1?
                
                # Let's interpret "reverse" as contrarian:
                # When bias becomes 1 (bullish), enter short (reverse the move)
                # When bias becomes -1 (bearish), enter long (reverse the move)
                
                # Actually, let's look at "reverse one sideded pdh == 1 trend sell entry"
                # This could mean: when pdh == 1, trend sell entry (short)
                # When pdh == -1, trend buy entry (long)
                
                # Let's implement this
                direction = 'short'
                entries.append({
                    'trade_num': trade_num,
                    'direction': direction,
                    'entry_ts': int(df.iloc[i]['time']),
                    'entry_time': datetime.fromtimestamp(df.iloc[i]['time'], tz=timezone.utc).isoformat(),
                    'entry_price_guess': df.iloc[i]['close'],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': df.iloc[i]['close'],
                    'raw_price_b': df.iloc[i]['close']
                })
                trade_num += 1
            elif df.iloc[i]['bias'] == -1:
                direction = 'long'
                entries.append({
                    'trade_num': trade_num,
                    'direction': direction,
                    'entry_ts': int(df.iloc[i]['time']),
                    'entry_time': datetime.fromtimestamp(df.iloc[i]['time'], tz=timezone.utc).isoformat(),
                    'entry_price_guess': df.iloc[i]['close'],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': df.iloc[i]['close'],
                    'raw_price_b': df.iloc[i]['close']
                })
                trade_num += 1
    
    return entries