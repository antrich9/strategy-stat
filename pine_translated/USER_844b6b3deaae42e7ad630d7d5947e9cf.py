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
    
    # Ensure we have enough data
    if len(df) < 3:
        return []
    
    # Helper functions to identify OB and FVG
    def is_up(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]
    
    def is_down(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]
    
    def is_ob_up(idx):
        # Bullish OB: previous bar is down, current bar is up, close > high of previous
        return (is_down(idx + 1) and is_up(idx) and 
                df['close'].iloc[idx] > df['high'].iloc[idx + 1])
    
    def is_ob_down(idx):
        # Bearish OB: previous bar is up, current bar is down, close < low of previous
        return (is_up(idx + 1) and is_down(idx) and 
                df['close'].iloc[idx] < df['low'].iloc[idx + 1])
    
    def is_fvg_up(idx):
        # Bullish FVG: low > high 2 bars back
        return df['low'].iloc[idx] > df['high'].iloc[idx + 2]
    
    def is_fvg_down(idx):
        # Bearish FVG: high < low 2 bars back
        return df['high'].iloc[idx] < df['low'].iloc[idx + 2]
    
    # Time window check (London time)
    # Morning: 7:45 - 9:45
    # Afternoon: 14:45 - 16:45
    def is_within_time_window(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        # Convert to London time (could be BST or GMT depending on date)
        # For simplicity, we'll use UTC and assume London is UTC+0 or UTC+1
        # Actually, let's try to handle this properly
        
        # Try to determine if it's BST (UTC+1) or GMT (UTC+0)
        # BST: March last Sunday to October last Sunday
        year = dt.year
        
        # Find last Sunday of March
        march_31 = datetime(year, 3, 31, tzinfo=timezone.utc)
        day_of_week = march_31.weekday()  # 0=Monday, 6=Sunday
        days_to_subtract = (day_of_week + 1) % 7
        last_sunday_march = march_31 - timedelta(days=days_to_subtract)
        
        # Find last Sunday of October
        oct_31 = datetime(year, 10, 31, tzinfo=timezone.utc)
        day_of_week = oct_31.weekday()
        days_to_subtract = (day_of_week + 1) % 7
        last_sunday_october = oct_31 - timedelta(days=days_to_subtract)
        
        current = dt
        if current >= last_sunday_march and current < last_sunday_october:
            # BST (UTC+1)
            london_hour = (dt.hour - 1) % 24
        else:
            # GMT (UTC+0)
            london_hour = dt.hour
        
        hour = london_hour
        minute = dt.minute
        
        # Morning window: 7:45 to 9:45
        morning_start = (7, 45)
        morning_end = (9, 45)
        
        # Afternoon window: 14:45 to 16:45
        afternoon_start = (14, 45)
        afternoon_end = (16, 45)
        
        def in_range(h, m, start, end):
            if start[0] < end[0] or (start[0] == end[0] and start[1] <= end[1]):
                return (h > start[0] or (h == start[0] and m >= start[1])) and \
                       (h < end[0] or (h == end[0] and m <= end[1]))
            else:
                return (h > start[0] or (h == start[0] and m >= start[1])) or \
                       (h < end[0] or (h == end[0] and m <= end[1]))
        
        return in_range(hour, minute, morning_start, morning_end) or \
               in_range(hour, minute, afternoon_start, afternoon_end)
    
    # Calculate OB and FVG conditions
    # obUp = isObUp(1) means check at index i where i-1 is the OB bar
    # So for bar i, we check if bar i-1 is OB up and bar i is FVG up
    # Actually, let's think about this differently
    
    # Pine Script:
    # obUp = isObUp(1)  // This is evaluated at current bar, checks if bar 1 back is OB up
    # fvgUp = isFvgUp(0) // This is evaluated at current bar, checks if current bar is FVG up
    
    # So for an entry at bar i, we need:
    # - At bar i-1: isObUp(1) was true (which means bar i is the current bar in Pine's perspective? No)
    
    # Let's trace through:
    # isObUp(index) checks: isDown(index + 1) and isUp(index) and close[index] > high[index + 1]
    # isObUp(1) checks: isDown(2) and isUp(1) and close[1] > high[2]
    # So isObUp(1) is true when the bar at index 1 is a bullish candle that exceeds the high of the bar at index 2, and the bar at index 2 is bearish.
    
    # In the strategy, obUp is calculated once at the start. But in a backtester, we need to calculate it for each bar.
    
    # Actually, looking at the Pine Script, these are calculated once at the start:
    # obUp = isObUp(1)
    # This means on every bar, obUp has the same value (calculated from the current bar looking back).
    
    # Wait, that's not right. In Pine Script, `obUp = isObUp(1)` is evaluated on each bar. On bar i, `isObUp(1)` checks:
    # - isDown(2): close[i-2] < open[i-2]
    # - isUp(1): close[i-1] > open[i-1]
    # - close[i-1] > high[i-2]
    # So it's checking if the previous bar (index 1 from current) is a bullish OB.
    
    # So for entry logic, on bar i:
    # - Check if obUp is true (which means bar i-1 is a bullish OB)
    # - Check if fvgUp is true (which means bar i is a bullish FVG)
    # - If both, enter long
    
    # So the entry condition for long is: obUp[i] and fvgUp[i]
    # where obUp[i] = isObUp(1) at bar i = isDown(i+1) and isUp(i) and close[i] > high[i+1]
    # Wait, this is confusing.
    
    # Let me re-read: isObUp(index) =>
    #     isDown(index + 1) and isUp(index) and close[index] > high[index + 1]
    
    # So isObUp(1):
    #     isDown(2) and isUp(1) and close[1] > high[2]
    
    # At bar i (current bar), obUp = isObUp(1):
    #     isDown(i+1) and isUp(i) and close[i] > high[i+1]
    
    # Wait, close[1] at bar i means close of bar i? No.
    # In Pine Script, when you write `close[1]` at bar i, it means the close of bar i-1.
    # So `close[1]` at bar i is close of bar i-1.
    # And `high[2]` at bar i is high of bar i-2.
    
    # So isObUp(1) at bar i:
    #     isDown(i+1) and isUp(i) and close(i) > high(i+1)
    # No wait, `index` in the function is relative to the current bar.
    
    # Let's say we're at bar i (the current bar).
    # We call isObUp(1).
    # Inside the function, `index` is 1.
    # So:
    # - isDown(index + 1) = isDown(2) = close[i-2] < open[i-2] (2 bars back)
    # - isUp(index) = isUp(1) = close[i-1] > open[i-1] (previous bar)
    # - close[index] = close[i-1] (previous bar)
    # - high[index + 1] = high[i] (current bar? No, high[2] at bar i is high of bar i-2)
    
    # Let's be very careful:
    # In Pine Script, `high[1]` at bar i refers to the high of bar i-1 (previous bar).
    # So `high[2]` at bar i refers to the high of bar i-2.
    
    # So isObUp(1) at bar i:
    # - isDown(2): checks if bar i-2 is down (close < open)
    # - isUp(1): checks if bar i-1 is up (close > open)
    # - close[1]: close of bar i-1
    # - high[2]: high of bar i-2
    # - Condition: close[i-1] > high[i-2]
    
    # So obUp at bar i is true when:
    # - Bar i-2 is bearish
    # - Bar i-1 is bullish
    # - Close of bar i-1 > high of bar i-2
    
    # And fvgUp at bar i (isFvgUp(0)):
    # isFvgUp(index) => (low[index] > high[index + 2])
    # isFvgUp(0) at bar i:
    # - low[0]: low of bar i (current bar)
    # - high[2]: high of bar i-2
    # - Condition: low[i] > high[i-2]
    
    # So for a long entry at bar i, we need obUp and fvgUp at bar i:
    # - Bar i-2 is bearish
    # - Bar i-1 is bullish with close > high of i-2
    # - Bar i (current) has low > high of i-2
    
    # This makes sense for a "stacked" setup where the FVG is above the OB.
    
    # Similarly for short:
    # isObDown(1) at bar i:
    # - isUp(2): bar i-2 is bullish
    # - isDown(1): bar i-1 is bearish
    # - close[i-1] < low[i-2]
    
    # isFvgDown(0) at bar i:
    # - high[i] < low[i-2]
    
    # So for short: bar i-2 bullish, bar i-1 bearish with close < low of i-2, and bar i high < low of i-2.
    
    # Perfect. Now I can implement this.
    
    # Need to calculate these for each bar
    n = len(df)
    
    # Initialize OB and FVG arrays
    ob_up = np.zeros(n, dtype=bool)
    ob_down = np.zeros(n, dtype=bool)
    fvg_up = np.zeros(n, dtype=bool)
    fvg_down = np.zeros(n, dtype=bool)
    
    # Calculate from index 2 onwards (need 2 bars back)
    for i in range(2, n):
        # obUp at bar i: isObUp(1) at bar i
        # isDown(i+1) = isDown(2) at bar i: df['close'].iloc[i-2] < df['open'].iloc[i-2]
        # isUp(i) = isUp(1) at bar i: df['close'].iloc[i-1] > df['open'].iloc[i-1]
        # close[i-1] > high[i-2]: df['close'].iloc[i-1] > df['high'].iloc[i-2]
        
        ob_up[i] = (df['close'].iloc[i-2] < df['open'].iloc[i-2] and  # isDown(i+1)
                    df['close'].iloc[i-1] > df['open'].iloc[i-1] and   # isUp(i)
                    df['close'].iloc[i-1] > df['high'].iloc[i-2])      # close[i-1] > high[i-2]
        
        # obDown at bar i: isObDown(1) at bar i
        # isUp(i+1): df['close'].iloc[i-2] > df['open'].iloc[i-2]
        # isDown(i): df['close'].iloc[i-1] < df['open'].iloc[i-1]
        # close[i-1] < low[i-2]: df['close'].iloc[i-1] < df['low'].iloc[i-2]
        
        ob_down[i] = (df['close'].iloc[i-2] > df['open'].iloc[i-2] and  # isUp(i+1)
                      df['close'].iloc[i-1] < df['open'].iloc[i-1] and   # isDown(i)
                      df['close'].iloc[i-1] < df['low'].iloc[i-2])       # close[i-1] < low[i-2]
        
        # fvgUp at bar i: isFvgUp(0) at bar i
        # low[i] > high[i-2]
        fvg_up[i] = df['low'].iloc[i] > df['high'].iloc[i-2]
        
        # fvgDown at bar i: isFvgDown(0) at bar i
        # high[i] < low[i-2]
        fvg_down[i] = df['high'].iloc[i] < df['low'].iloc[i-2]
    
    # Time window check
    time_window = np.zeros(n, dtype=bool)
    for i in range(n):
        time_window[i] = is_within_time_window(df['time'].iloc[i])
    
    # Entry signals
    long_signal = ob_up & fvg_up & time_window
    short_signal = ob_down & fvg_down & time_window
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(n):
        if long_signal.iloc[i] if isinstance(long_signal, pd.Series) else long_signal[i]:
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif short_signal.iloc[i] if isinstance(short_signal, pd.Series) else short_signal[i]:
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries