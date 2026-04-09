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
    atr_period = 14
    
    # Calculate True Range
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - prev_close)
    tr3 = abs(df['low'] - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder ATR
    atr = tr.ewm(alpha=1.0/atr_period, adjust=False).mean()
    
    # Initialize state variables
    ph = np.nan  # previous high (high of previous day)
    pl = np.nan  # previous low (low of previous day)
    ch = np.nan  # current high (high of current day)
    cl = np.nan  # current low (low of current day)
    
    trade_active = False
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        ts = df['time'].iloc[i]
        date_curr = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        
        # Determine if this is a new day
        if i == 0:
            new_day = True
        else:
            ts_prev = df['time'].iloc[i-1]
            date_prev = datetime.fromtimestamp(ts_prev, tz=timezone.utc).date()
            new_day = date_curr > date_prev
        
        # Entry condition: previous close > previous high (ph)
        if not pd.isna(atr.iloc[i]) and not trade_active:
            if i > 0:
                close_prev = df['close'].iloc[i-1]
                if not pd.isna(ph) and close_prev > ph:
                    entry_price = df['close'].iloc[i]
                    entry_ts = df['time'].iloc[i]
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': entry_ts,
                        'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                        'entry_price_guess': entry_price,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': entry_price,
                        'raw_price_b': entry_price
                    })
                    trade_num += 1
                    trade_active = True
        
        # Handle new day transition (update ph, pl, ch, cl)
        if new_day:
            # At the start of a new day, ph/pl get values from previous day
            if not pd.isna(ch):
                ph = ch
                pl = cl
            # Initialize ch/cl for new day
            ch = df['high'].iloc[i]
            cl = df['low'].iloc[i]
        else:
            # Within same day, update running high/low
            if pd.isna(ch):
                ch = df['high'].iloc[i]
            else:
                ch = max(ch, df['high'].iloc[i])
            if pd.isna(cl):
                cl = df['low'].iloc[i]
            else:
                cl = min(cl, df['low'].iloc[i])
        
        # Reset trade_active if no entry triggered
        if not trade_active:
            trade_active = False
    
    return entries