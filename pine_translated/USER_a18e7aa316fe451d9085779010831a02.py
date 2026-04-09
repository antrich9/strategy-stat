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
    
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Convert unix timestamp to datetime for time filtering
    data['datetime'] = pd.to_datetime(data['time'], unit='s', utc=True)
    data['hour'] = data['datetime'].dt.hour
    data['minute'] = data['datetime'].dt.minute
    data['dayofweek'] = data['datetime'].dt.dayofweek  # Monday=0, Friday=4
    
    # Time windows (London timezone)
    # Morning: 08:00 - 09:55 (hour=8 or hour=9 with minute<=55)
    isWithinMorningWindow = ((data['hour'] == 8) | ((data['hour'] == 9) & (data['minute'] <= 55)))
    
    # Afternoon: 14:00 - 16:55 (hour=14,15 or hour=16 with minute<=55)
    isWithinAfternoonWindow = ((data['hour'] == 14) | (data['hour'] == 15) | ((data['hour'] == 16) & (data['minute'] <= 55)))
    
    # Friday morning exclusion
    isFridayMorningWindow = (data['dayofweek'] == 4) & (data['hour'] == 8)
    
    # Combined time window condition
    in_trading_window = isWithinMorningWindow | isWithinAfternoonWindow
    
    # Calculate EMAs
    ema200 = data['close'].ewm(span=200, adjust=False).mean()
    ema50 = data['close'].ewm(span=50, adjust=False).mean()
    ema20 = data['close'].ewm(span=20, adjust=False).mean()
    
    # Higher timeframe EMA (240 minutes = 4 hours)
    # Approximate using span=40 on 15min data
    higherTimeFrameEMA = data['close'].ewm(span=40, adjust=False).mean()
    
    # KDJ Indicator
    # Stochastic %K with 14 period
    lowest_low = data['low'].rolling(window=14).min()
    highest_high = data['high'].rolling(window=14).max()
    stoch_k = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
    stoch_k = stoch_k.replace([np.inf, -np.inf], np.nan)
    
    # Smooth %K with 3 period SMA to get KDJ K
    kdj_k = stoch_k.rolling(window=3).mean()
    # Smooth K with 3 period SMA to get KDJ D
    kdj_d = kdj_k.rolling(window=3).mean()
    # KDJ J = 3*K - 2*D
    kdj_j = 3 * kdj_k - 2 * kdj_d
    
    # Wilder RSI (14 period)
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsiValue = 100 - (100 / (1 + rs))
    rsiValue = rsiValue.replace([np.inf, -np.inf], np.nan)
    
    # Build entry conditions as boolean series
    # KDJ crossover (K crosses above D)
    kdj_crossover = (kdj_k > kdj_d) & (kdj_k.shift(1) <= kdj_d.shift(1))
    
    # KDJ crossunder (K crosses below D)
    kdj_crossunder = (kdj_k < kdj_d) & (kdj_k.shift(1) >= kdj_d.shift(1))
    
    # Long condition: KDJ crossover AND close > all EMAs AND RSI > 50 AND close > higherTF EMA
    longCondition = (
        kdj_crossover & 
        (data['close'] > ema200) & 
        (data['close'] > ema50) & 
        (data['close'] > ema20) & 
        (rsiValue > 50) & 
        (data['close'] > higherTimeFrameEMA)
    )
    
    # Short condition: KDJ crossunder AND close < all EMAs AND RSI < 50 AND close < higherTF EMA
    shortCondition = (
        kdj_crossunder & 
        (data['close'] < ema200) & 
        (data['close'] < ema50) & 
        (data['close'] < ema20) & 
        (rsiValue < 50) & 
        (data['close'] < higherTimeFrameEMA)
    )
    
    # Combine with time window and position check
    long_signal = in_trading_window & ~isFridayMorningWindow & longCondition
    short_signal = in_trading_window & ~isFridayMorningWindow & shortCondition
    
    # Track position state to avoid entries while in position
    in_position = False
    
    entries = []
    trade_num = 1
    
    for i in range(len(data)):
        # Check if we can enter (not in position, valid time window, not Friday morning)
        can_enter = not in_position and in_trading_window.iloc[i] and not isFridayMorningWindow.iloc[i]
        
        if can_enter:
            entry_price = data['close'].iloc[i]
            entry_ts = int(data['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
            if long_signal.iloc[i]:
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
                in_position = True
                trade_num += 1
            elif short_signal.iloc[i]:
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
                in_position = True
                trade_num += 1
        else:
            # Reset position flag if we're not in a valid window or other conditions
            # This is a simplification - real position tracking would need exit logic
            if not (in_trading_window.iloc[i] & ~isFridayMorningWindow.iloc[i]):
                in_position = False
    
    return entries