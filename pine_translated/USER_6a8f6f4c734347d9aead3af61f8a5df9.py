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
    
    # Initialize columns
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Get previous day high and low using security-like approach
    df['prev_day_high'] = df['high'].shift(1).rolling(window=2, min_periods=1).max().shift(1)
    df['prev_day_low'] = df['low'].shift(1).rolling(window=2, min_periods=1).min().shift(1)
    
    # Detect new day
    df['is_new_day'] = df['time'].dt.date != df['time'].dt.date.shift(1)
    df.loc[df['is_new_day'].fillna(True), 'is_new_day'] = True
    
    # Flags for PDH/PDL sweep - initialize as False
    df['flagpdh'] = False
    df['flagpdl'] = False
    df['waitingForEntry'] = False
    df['waitingForShortEntry1'] = False
    
    # Calculate flags for PDH/PDL sweep
    for i in range(1, len(df)):
        if df.loc[i, 'close'] > df.loc[i, 'prev_day_high']:
            df.loc[i, 'flagpdh'] = True
        if df.loc[i, 'close'] < df.loc[i, 'prev_day_low']:
            df.loc[i, 'flagpdl'] = True
    
    # OB and FVG conditions
    def is_up(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]
    
    def is_down(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]
    
    # Store OB and FVG as series
    ob_up = pd.Series(False, index=df.index)
    ob_down = pd.Series(False, index=df.index)
    fvg_up = pd.Series(False, index=df.index)
    fvg_down = pd.Series(False, index=df.index)
    
    for i in range(2, len(df)):
        try:
            # OB conditions
            ob_up.iloc[i] = (df['close'].iloc[i] > df['open'].iloc[i] and 
                           df['close'].iloc[i-1] < df['open'].iloc[i-1] and 
                           df['close'].iloc[i] > df['high'].iloc[i-1])
            
            ob_down.iloc[i] = (df['close'].iloc[i] < df['open'].iloc[i] and 
                              df['close'].iloc[i-1] > df['open'].iloc[i-1] and 
                              df['close'].iloc[i] < df['low'].iloc[i-1])
            
            # FVG conditions
            fvg_up.iloc[i] = df['low'].iloc[i] > df['high'].iloc[i+2] if i+2 < len(df) else False
            fvg_down.iloc[i] = df['high'].iloc[i] < df['low'].iloc[i+2] if i+2 < len(df) else False
        except:
            pass
    
    # Reset waiting flags on new day
    for i in range(1, len(df)):
        if df['is_new_day'].iloc[i]:
            df.loc[df.index[i], 'waitingForEntry'] = False
            df.loc[df.index[i], 'waitingForShortEntry1'] = False
    
    # Entry logic - Bullish (Long)
    long_entry = pd.Series(False, index=df.index)
    short_entry = pd.Series(False, index=df.index)
    
    in_long = False
    in_short = False
    waiting_for_long = False
    waiting_for_short = False
    
    # Fib level calculation helpers
    tracking_high_after_bearish = np.nan
    tracking_low_after_bullish = np.nan
    fib618 = np.nan
    
    for i in range(3, len(df)):
        current_bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        
        # Reset on new day
        if current_bar['is_new_day']:
            in_long = False
            in_short = False
            waiting_for_long = False
            waiting_for_short = False
            tracking_high_after_bearish = np.nan
            tracking_low_after_bullish = np.nan
            fib618 = np.nan
        
        # Check if bullish liquidity (PDH sweep) just happened - track high
        if current_bar['flagpdh'] and not prev_bar['flagpdh']:
            tracking_high_after_bearish = df['high'].iloc[i]
            # Find highest high since bearish liquidity
            for j in range(i+1, min(i+50, len(df))):
                if df['high'].iloc[j] > tracking_high_after_bearish:
                    tracking_high_after_bearish = df['high'].iloc[j]
                if df['flagpdl'].iloc[j]:  # If bearish liquidity found, stop tracking
                    break
            # Calculate 61.8% fib from high to recent low
            recent_low = df['low'].iloc[i:min(i+20, len(df))].min()
            if not np.isnan(tracking_high_after_bearish) and not np.isnan(recent_low):
                fib618 = tracking_high_after_bearish - 0.618 * (tracking_high_after_bearish - recent_low)
        
        # Check if bearish liquidity (PDL sweep) just happened - track low
        if current_bar['flagpdl'] and not prev_bar['flagpdl']:
            tracking_low_after_bullish = df['low'].iloc[i]
            # Find lowest low since bullish liquidity
            for j in range(i+1, min(i+50, len(df))):
                if df['low'].iloc[j] < tracking_low_after_bullish:
                    tracking_low_after_bullish = df['low'].iloc[j]
                if df['flagpdh'].iloc[j]:  # If bullish liquidity found, stop tracking
                    break
            # Calculate 61.8% fib from low to recent high
            recent_high = df['high'].iloc[i:min(i+20, len(df))].max()
            if not np.isnan(tracking_low_after_bullish) and not np.isnan(recent_high):
                fib618 = tracking_low_after_bullish + 0.618 * (recent_high - tracking_low_after_bullish)
        
        # Long entry logic - OB+FVG stacked + waiting for fib level touch
        if current_bar['flagpdh']:  # PDH was swept (bullish context)
            if ob_up.iloc[i] and fvg_up.iloc[i]:
                waiting_for_long = True
        
        # Short entry logic - OB+FVG stacked + waiting for fib level touch
        if current_bar['flagpdl']:  # PDL was swept (bearish context)
            if ob_down.iloc[i] and fvg_down.iloc[i]:
                waiting_for_short = True
        
        # Trigger long entry when price approaches fib level
        if waiting_for_long and not in_long:
            if not np.isnan(fib618) and current_bar['low'] <= fib618:
                long_entry.iloc[i] = True
                waiting_for_long = False
                in_long = True
        
        # Trigger short entry when price approaches fib level
        if waiting_for_short and not in_short:
            if not np.isnan(fib618) and current_bar['high'] >= fib618:
                short_entry.iloc[i] = True
                waiting_for_short = False
                in_short = True
    
    # Build result
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_entry.iloc[i]:
            entry_price = df['close'].iloc[i]
            ts = int(df['time'].iloc[i].timestamp())
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
        
        if short_entry.iloc[i]:
            entry_price = df['close'].iloc[i]
            ts = int(df['time'].iloc[i].timestamp())
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
    
    return entries