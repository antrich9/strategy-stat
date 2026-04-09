import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    atr_period = 14
    
    # Calculate True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder ATR
    atr = tr.ewm(alpha=1.0/atr_period, adjust=False).mean()
    
    # Detect new days
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['day'] = df['datetime'].dt.date
    df['new_day'] = df['day'] != df['day'].shift(1)
    df['new_day'].iloc[0] = True  # First bar is a "new day" for initialization
    
    # Initialize tracking variables
    ph = np.nan
    pl = np.nan
    ch = np.nan
    cl = np.nan
    co = np.nan
    
    entries = []
    trade_num = 1
    current_position = "flat"
    
    # Build boolean conditions first
    close_prev = df['close'].shift(1)
    
    # Entry condition: close[1] < pl and flat
    long_condition = (close_prev < pl) & (current_position == "flat")
    
    for i in range(len(df)):
        if i == 0:
            continue
            
        if df['new_day'].iloc[i]:
            if not np.isnan(ch):
                ph = ch
                pl = cl
            ch = df['high'].iloc[i]
            cl = df['low'].iloc[i]
            co = df['open'].iloc[i]
            # Reset position at start of new day if it was open
            if current_position != "flat":
                current_position = "flat"
        else:
            ch = max(ch, df['high'].iloc[i])
            cl = min(cl, df['low'].iloc[i])
        
        # Long entry condition: previous close < previous day's low (pl)
        if not np.isnan(pl) and not np.isnan(atr.iloc[i]):
            prev_close = df['close'].iloc[i-1] if i > 0 else np.nan
            if current_position == "flat" and prev_close < pl:
                entry_price_guess = df['close'].iloc[i]
                entry_ts = int(df['time'].iloc[i])
                
                entry = {
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price_guess,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price_guess,
                    'raw_price_b': entry_price_guess
                }
                entries.append(entry)
                trade_num += 1
                current_position = "long"
    
    return entries