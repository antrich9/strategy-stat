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
    
    # Ensure required columns exist
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            return []
    
    df = df.copy()
    n = len(df)
    
    if n < 5:
        return []
    
    # Helper functions for OB and FVG detection
    def is_up(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]
    
    def is_down(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]
    
    # Compute OB and FVG conditions as boolean arrays
    # obUp(index): isDown(index + 1) and isUp(index) and close[index] > high[index + 1]
    # For bar i (0-indexed), obUp at i means: isDown(i+1), isUp(i), close_i > high_{i+1}
    # This means we look at bar i and bar i+1
    
    # Let's compute for each bar i
    # obUp[i] means the condition is true when we're at bar i (looking at i and i+1)
    # But in Pine: obUp = isObUp(1) means checking bar index 1 (current bar - 1) and bar index 2
    
    # For 0-indexed Python:
    # isObUp(i) checks: isDown(i+1), isUp(i), close[i] > high[i+1]
    # So for bar i in Python (0-indexed), this checks current bar i and next bar i+1
    
    # obUp = isObUp(1) in Pine means checking from bar 1 onwards
    # In Python 0-indexed: for i >= 1, check bars i and i+1
    
    # Let's create shifted arrays for efficient computation
    close = df['close'].values
    open_arr = df['open'].values
    high = df['high'].values
    low = df['low'].values
    
    # For obUp: isDown(i+1) and isUp(i) and close[i] > high[i+1]
    # We need close_i > high_{i+1}
    # shift(-1) means next bar
    close_shifted_next = np.roll(close, -1)  # shift -1 = next bar
    high_shifted_next = np.roll(high, -1)
    low_shifted_next = np.roll(low, -1)
    
    close_shifted_prev = np.roll(close, 1)  # shift 1 = previous bar
    high_shifted_prev = np.roll(high, 1)
    low_shifted_prev = np.roll(low, 1)
    
    close_shifted_2 = np.roll(close, 2)  # 2 bars ago
    high_shifted_2 = np.roll(high, 2)
    low_shifted_2 = np.roll(low, 2)
    
    close_shifted_3 = np.roll(close, 3)
    high_shifted_3 = np.roll(high, 3)
    low_shifted_3 = np.roll(low, 3)
    
    # Set last elements to NaN since we can't look ahead
    close_shifted_next[-1] = np.nan
    high_shifted_next[-1] = np.nan
    low_shifted_next[-1] = np.nan
    close_shifted_prev[0] = np.nan
    high_shifted_prev[0] = np.nan
    low_shifted_prev[0] = np.nan
    close_shifted_2[:2] = np.nan
    high_shifted_2[:2] = np.nan
    low_shifted_2[:2] = np.nan
    close_shifted_3[:3] = np.nan
    high_shifted_3[:3] = np.nan
    low_shifted_3[:3] = np.nan
    
    is_up_arr = close > open_arr
    is_down_arr = close < open_arr
    
    # obUp(i): isDown(i+1) and isUp(i) and close_i > high_{i+1}
    # In Pine: isObUp(1) means check from bar index 1
    # For Python 0-indexed with shift: bar i checks i and i+1
    # obUp_series[i] = True if at bar i we have: down at i+1, up at i, close_i > high_{i+1}
    ob_up = (is_down_arr[1:] if len(is_down_arr) > 1 else np.array([])).astype(bool)
    ob_up_full = np.zeros(n, dtype=bool)
    if n > 1:
        # For i from 0 to n-2: check isDown(i+1), isUp(i), close[i] > high[i+1]
        for i in range(n - 1):
            if i + 1 < n:
                if is_down_arr[i + 1] and is_up_arr[i] and close[i] > high[i + 1]:
                    ob_up_full[i] = True
    
    # obDown(i): isUp(i+1) and isDown(i) and close_i < low_{i+1}
    ob_down = np.zeros(n, dtype=bool)
    if n > 1:
        for i in range(n - 1):
            if i + 1 < n:
                if is_up_arr[i + 1] and is_down_arr[i] and close[i] < low[i + 1]:
                    ob_down_full[i] = True
    
    # fvgUp(index): low[index] > high[index + 2]
    # For fvgUp(0): check low_0 > high_2
    fvg_up = np.zeros(n, dtype=bool)
    for i in range(n):
        if i + 2 < n:
            if low[i] > high[i + 2]:
                fvg_up[i] = True
    
    # fvgDown(index): high[index] < low[index + 2]
    fvg_down = np.zeros(n, dtype=bool)
    for i in range(n):
        if i + 2 < n:
            if high[i] < low[i + 2]:
                fvg_down[i] = True
    
    # Wilder RSI implementation
    def wilder_rsi(close_series, period=14):
        delta = close_series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilder ATR implementation
    def wilder_atr(high_series, low_series, close_series, period=20):
        high_low = high_series - low_series
        high_close = (high_series - close_series.shift(1)).abs()
        low_close = (low_series - close_series.shift(1)).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        atr = true_range.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    close_series = df['close']
    high_series = df['high']
    low_series = df['low']
    volume_series = df['volume']
    
    # Indicators
    rsi_val = wilder_rsi(close_series, 14)
    atr_val = wilder_atr(high_series, low_series, close_series, 20)
    sma_volume = volume_series.rolling(9).mean()
    loc = close_series.ewm(span=54, adjust=False).mean()
    
    # Filters
    # volfilt = inp1 ? volume[1] > ta.sma(volume, 9)*1.5 : true
    # For bar i, volume[1] means previous bar
    volfilt = np.ones(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(sma_volume.iloc[i-1]) and sma_volume.iloc[i-1] > 0:
            if volume_series.iloc[i-1] <= sma_volume.iloc[i-1] * 1.5:
                volfilt[i] = False
    
    # atr = ta.atr(20) / 1.5
    # atrfilt = inp2 ? ((low - high[2] > atr) or (low[2] - high > atr)) : true
    atrfilt = np.ones(n, dtype=bool)
    for i in range(n):
        if i >= 2 and not np.isnan(atr_val.iloc[i]):
            if not ((low[i] - high[i-2] > atr_val.iloc[i]) or (low[i-2] - high[i] > atr_val.iloc[i])):
                if inp2:  # ATR filter enabled
                    atrfilt[i] = False
        elif i < 2:
            atrfilt[i] = True  # Can't compute, assume pass
    
    # loc2 = loc > loc[1]
    loc2 = (loc > loc.shift(1)).values
    
    # locfiltb = inp3 ? loc2 : true
    # locfilts = inp3 ? not loc2 : true
    locfiltb = np.ones(n, dtype=bool)
    locfilts = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isnan(loc2[i]) and not np.isnan(loc2[i-1] if i > 0 else False):
            if loc2[i]:
                locfiltb[i] = True
                locfilts[i] = False
            else:
                locfiltb[i] = False
                locfilts[i] = True
    
    # Time window for London time (7:45-9:45 and 14:45-16:45)
    in_trading_window = np.zeros(n, dtype=bool)
    
    for i in range(n):
        ts = df['time'].iloc[i]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        
        # Morning window: 7:45 to 9:45
        morning_start = (hour == 7 and minute >= 45) or (hour > 7)
        morning_end = (hour < 9) or (hour == 9 and minute < 45)
        
        # Afternoon window: 14:45 to 16:45
        afternoon_start = (hour == 14 and minute >= 45) or (hour > 14)
        afternoon_end = (hour < 16) or (hour == 16 and minute < 45)
        
        if morning_start and morning_end:
            in_trading_window[i] = True
        elif afternoon_start and afternoon_end:
            in_trading_window[i] = True
    
    # Entry signals
    # Long entry: obUp and fvgUp with all filters and time window
    # Short entry: obDown and fvgDown with all filters and time window
    
    entries = []
    trade_num = 1
    
    for i in range(n):
        if i < 2:
            continue
        if np.isnan(atr_val.iloc[i]):
            continue
        
        is_long = (ob_up[i] and fvg_up[i] and volfilt[i] and atrfilt[i] and 
                   locfiltb[i] and in_trading_window[i])
        is_short = (ob_down[i] and fvg_down[i] and volfilt[i] and atrfilt[i] and 
                     locfilts[i] and in_trading_window[i])
        
        direction = None
        if is_long and not is_short:
            direction = 'long'
        elif is_short and not is_long:
            direction = 'short'
        elif is_long and is_short:
            direction = 'long'  # Prefer long if both
        else:
            continue
        
        entry_price = df['close'].iloc[i]
        entry_ts = df['time'].iloc[i]
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        
        entries.append({
            'trade_num': trade_num,
            'direction': direction,
            'entry_ts': int(entry_ts),
            'entry_time': entry_time,
            'entry_price_guess': float(entry_price),
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': float(entry_price),
            'raw_price_b': float(entry_price)
        })
        
        trade_num += 1
    
    return entries