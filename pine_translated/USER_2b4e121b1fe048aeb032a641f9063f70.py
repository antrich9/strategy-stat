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
    
    # Ensure we have the required columns
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        return []
    
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Convert time to datetime for time-based filtering
    data['datetime'] = pd.to_datetime(data['time'], unit='ms', utc=True)
    data['hour'] = data['datetime'].dt.hour
    data['minute'] = data['datetime'].dt.minute
    
    # Define trading windows
    # Window 1: 07:00 - 10:59
    # Window 2: 15:00 - 16:59
    in_window_1 = (data['hour'] >= 7) & (data['hour'] < 11)
    in_window_2 = (data['hour'] >= 15) & (data['hour'] < 17)
    in_trading_window = in_window_1 | in_window_2
    
    # Calculate SMA for trend filter (loc = ta.sma(close, 54))
    data['loc'] = data['close'].rolling(54).mean()
    data['loc_prev'] = data['loc'].shift(1)
    loc_trend_up = data['loc'] > data['loc_prev']
    loc_trend_down = data['loc'] <= data['loc_prev']
    
    # Calculate volume filter (volfilt = volume[1] > ta.sma(volume, 9) * 1.5)
    sma_vol = data['volume'].rolling(9).mean()
    data['vol_filt'] = data['volume'].shift(1) > sma_vol * 1.5
    
    # Calculate ATR for ATR filter (atr = ta.atr(20) / 1.5)
    # Wilder ATR implementation
    high = data['high']
    low = data['low']
    close = data['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/20, adjust=False).mean() / 1.5
    data['atr_filt'] = ((low - high.shift(2) > atr) | (low.shift(2) - high > atr))
    
    # Calculate FVG conditions
    # Bullish FVG: low > high[2]
    bfvg = low > high.shift(2)
    # Bearish FVG: high < low[2]
    sfvg = high < low.shift(2)
    
    # Calculate Previous Day High/Low (using daily aggregation)
    data['day'] = data['datetime'].dt.date
    daily_agg = data.groupby('day').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'open': 'first'
    }).reset_index()
    daily_agg['prev_day_high'] = daily_agg['high'].shift(1)
    daily_agg['prev_day_low'] = daily_agg['low'].shift(1)
    daily_agg = daily_agg.dropna()
    
    # Merge daily data back
    data = data.merge(daily_agg[['day', 'prev_day_high', 'prev_day_low']], on='day', how='left')
    data['prev_day_high'] = data['prev_day_high'].ffill()
    data['prev_day_low'] = data['prev_day_low'].ffill()
    
    # Calculate Current Day High/Low (using 240min aggregation)
    # For simplicity, we'll use rolling window approach for intraday
    data['curr_day_h'] = data['high'].rolling(window=24, min_periods=1).max()  # Approximation
    data['curr_day_l'] = data['low'].rolling(window=24, min_periods=1).min()  # Approximation
    
    # Entry conditions
    # Long entry: prev_day_high sweep + bullish FVG + stacked OB + trend filter + filters
    # Short entry: prev_day_low sweep + bearish FVG + stacked OB + trend filter + filters
    
    # Detect sweep of previous day high (close crosses above prev_day_high)
    data['sweep_prev_high'] = (data['close'] > data['prev_day_high']) & (data['close'].shift(1) <= data['prev_day_high'])
    # Detect sweep of previous day low (close crosses below prev_day_low)
    data['sweep_prev_low'] = (data['close'] < data['prev_day_low']) & (data['close'].shift(1) >= data['prev_day_low'])
    
    # Stacked OB conditions (simplified - OB is where price reversed)
    # For bullish OB: isDown(index + 1) and isUp(index) and close[index] > high[index + 1]
    is_down_prev = data['close'].shift(1) < data['open'].shift(1)
    is_up_curr = data['close'] >= data['open']
    ob_up = is_down_prev & is_up_curr & (data['close'] > data['high'].shift(1))
    
    # For bearish OB: isUp(index + 1) and isDown(index) and close[index] < low[index + 1]
    is_up_prev = data['close'].shift(1) >= data['open'].shift(1)
    is_down_curr = data['close'] < data['open']
    ob_down = is_up_prev & is_down_curr & (data['close'] < data['low'].shift(1))
    
    # FVG detection (more specific)
    # Bullish FVG up: low > high[2]
    fvg_up = low > high.shift(2)
    # Bearish FVG down: high < low[2]
    fvg_down = high < low.shift(2)
    
    # Combined long entry condition
    long_entry = (
        data['in_trading_window'] if 'in_trading_window' in data.columns else in_trading_window
    )
    long_entry = (
        data['sweep_prev_high'] & 
        fvg_up & 
        ob_up & 
        loc_trend_up & 
        data['vol_filt'] & 
        data['atr_filt']
    )
    
    # Combined short entry condition
    short_entry = (
        data['sweep_prev_low'] & 
        fvg_down & 
        ob_down & 
        loc_trend_down & 
        data['vol_filt'] & 
        data['atr_filt']
    )
    
    # Build list of entries
    entries = []
    trade_num = 1
    
    # Process long entries
    for i in range(len(data)):
        if long_entry.iloc[i] if hasattr(long_entry, 'iloc') else long_entry[i]:
            ts = int(data['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(data['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(data['close'].iloc[i]),
                'raw_price_b': float(data['close'].iloc[i])
            })
            trade_num += 1
    
    # Process short entries
    for i in range(len(data)):
        if short_entry.iloc[i] if hasattr(short_entry, 'iloc') else short_entry[i]:
            ts = int(data['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(data['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(data['close'].iloc[i]),
                'raw_price_b': float(data['close'].iloc[i])
            })
            trade_num += 1
    
    return entries