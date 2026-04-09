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
    close = df['close']
    high = df['high']
    low = df['low']
    time_col = df['time']
    n = len(df)
    
    if n < 5:
        return []
    
    # Inputs from strategy
    PP = 5
    atrLength = 14
    atrMultiplier = 1.5
    
    # Calculate Wilder RSI manually
    def wilder_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate Wilder ATR manually
    def wilder_atr(high, low, close, period):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    # Calculate 55-period ATR for ZigZag
    atr55 = wilder_atr(high, low, close, 55)
    
    # Calculate pivot highs and lows
    pivot_high = pd.Series(False, index=df.index)
    pivot_low = pd.Series(False, index=df.index)
    
    for i in range(PP, n - PP):
        is_high = True
        is_low = True
        for j in range(1, PP + 1):
            if high.iloc[i - j] >= high.iloc[i]:
                is_high = False
            if low.iloc[i - j] <= low.iloc[i]:
                is_low = False
        if is_high:
            pivot_high.iloc[i] = True
        if is_low:
            pivot_low.iloc[i] = True
    
    # Track major structure levels
    major_high = pd.Series(np.nan, index=df.index)
    major_low = pd.Series(np.nan, index=df.index)
    
    Bullish_Major_BoS = False
    Bearish_Major_BoS = False
    
    last_major_high = np.nan
    last_major_low = np.nan
    
    # Calculate ATR for stop loss
    atr14 = wilder_atr(high, low, close, atrLength)
    
    # Track structure
    for i in range(PP, n):
        if pivot_high.iloc[i]:
            current_high = high.iloc[i]
            if not np.isnan(last_major_high):
                if current_high > last_major_high:
                    Bullish_Major_BoS = True
                    Bearish_Major_BoS = False
            last_major_high = current_high
            major_high.iloc[i] = current_high
        
        if pivot_low.iloc[i]:
            current_low = low.iloc[i]
            if not np.isnan(last_major_low):
                if current_low < last_major_low:
                    Bearish_Major_BoS = True
                    Bullish_Major_BoS = False
            last_major_low = current_low
            major_low.iloc[i] = current_low
    
    # Fill forward for checking
    major_high = major_high.ffill()
    major_low = major_low.ffill()
    
    # Calculate RSI for filtering
    rsi = wilder_rsi(close, 14)
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(PP + 1, n - 1):
        if pd.isna(atr14.iloc[i]) or pd.isna(rsi.iloc[i]):
            continue
        if pd.isna(major_high.iloc[i]) or pd.isna(major_low.iloc[i]):
            continue
        
        mh = major_high.iloc[i]
        ml = major_low.iloc[i]
        rsi_val = rsi.iloc[i]
        
        # Long entry: price crosses above major high with bullish structure
        long_cond = (close.iloc[i] > mh and close.iloc[i-1] <= mh)
        
        # Short entry: price crosses below major low with bearish structure
        short_cond = (close.iloc[i] < ml and close.iloc[i-1] >= ml)
        
        if long_cond:
            direction = 'long'
            entry_price = close.iloc[i]
            entry_ts = int(time_col.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
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
        
        elif short_cond:
            direction = 'short'
            entry_price = close.iloc[i]
            entry_ts = int(time_col.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
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