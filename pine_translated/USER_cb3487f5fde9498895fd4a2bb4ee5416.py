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
    
    # Wilder ATR implementation
    def wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr
    
    # Wilder RSI implementation
    def wilder_rsi(close, length):
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Pivot detection functions
    def pivot_high(high, low, close, pp):
        ph = high.rolling(window=pp+1, center=False).max()
        ph_valid = ph.shift(1)
        for i in range(1, pp+1):
            ph_valid = ph_valid.where(high.shift(i) <= ph_valid, high.shift(i))
        return ph_valid
    
    def pivot_low(low_val, pp):
        pl = low_val.rolling(window=pp+1, center=False).min()
        pl_valid = pl.shift(1)
        for i in range(1, pp+1):
            pl_valid = pl_valid.where(low_val.shift(i) >= pl_valid, low_val.shift(i))
        return pl_valid
    
    # Initialize arrays for ZigZag and structure tracking
    n = len(df)
    pp = 5  # Pivot Period
    
    # Calculate ATR (length 55 from script)
    atr = wilder_atr(df['high'], df['low'], df['close'], 55)
    
    # Calculate pivots
    high_pivot = pivot_high(df['high'], df['low'], df['close'], pp)
    low_pivot = pivot_low(df['low'], pp)
    
    # Structure tracking variables
    major_high = pd.Series(np.nan, index=df.index)
    major_low = pd.Series(np.nan, index=df.index)
    minor_high = pd.Series(np.nan, index=df.index)
    minor_low = pd.Series(np.nan, index=df.index)
    
    bull_bos = pd.Series(False, index=df.index)
    bear_bos = pd.Series(False, index=df.index)
    bull_choch = pd.Series(False, index=df.index)
    bear_choch = pd.Series(False, index=df.index)
    
    bull_mss = pd.Series(False, index=df.index)
    bear_mss = pd.Series(False, index=df.index)
    
    # Detect market structure (simplified)
    # Major structure based on higher timeframe-like analysis
    for i in range(pp*3, n):
        if pd.notna(high_pivot.iloc[i]):
            # Check if this high breaks previous major high
            prev_highs = major_high.iloc[:i].dropna()
            if len(prev_highs) > 0:
                if df['high'].iloc[i] > prev_highs.iloc[-1]:
                    bull_bos.iloc[i] = True
            # Update major high
            if pd.isna(major_high.iloc[i]):
                major_high.iloc[i] = df['high'].iloc[i]
            elif df['high'].iloc[i] > major_high.iloc[i]:
                major_high.iloc[i] = df['high'].iloc[i]
        
        if pd.notna(low_pivot.iloc[i]):
            # Check if this low breaks previous major low
            prev_lows = major_low.iloc[:i].dropna()
            if len(prev_lows) > 0:
                if df['low'].iloc[i] < prev_lows.iloc[-1]:
                    bear_bos.iloc[i] = True
            # Update major low
            if pd.isna(major_low.iloc[i]):
                major_low.iloc[i] = df['low'].iloc[i]
            elif df['low'].iloc[i] < major_low.iloc[i]:
                major_low.iloc[i] = df['low'].iloc[i]
        
        # Detect ChoCh (Change of Character) - simplified
        if bull_bos.iloc[i] and i > 0 and bear_bos.iloc[i-1:i].any():
            bull_choch.iloc[i] = True
        if bear_bos.iloc[i] and i > 0 and bull_bos.iloc[i-1:i].any():
            bear_choch.iloc[i] = True
    
    # Calculate RSI for additional confirmation
    rsi = wilder_rsi(df['close'], 14)
    
    # Entry conditions based on the strategy logic
    # Long entry: Bullish market structure + RSI confirmation
    # Short entry: Bearish market structure + RSI confirmation
    
    long_condition = (bull_bos | bull_choch) & (rsi < 70)
    short_condition = (bear_bos | bear_choch) & (rsi > 30)
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(pp*3 + 1, n):
        # Skip if ATR is NaN
        if pd.isna(atr.iloc[i]):
            continue
        
        # Check long condition
        if long_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        
        # Check short condition
        if short_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries