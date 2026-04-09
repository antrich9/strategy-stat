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
    
    # Convert unix timestamp to datetime in UTC
    # The timestamps in df are in milliseconds (Pine Script uses milliseconds)
    # But looking at the Pine Script, time is in milliseconds: timestamp("Europe/London", ...)
    # Actually in Pine Script 5, time is usually in milliseconds
    # But the user says time is unix ts, so I assume it's seconds
    # Let's check: Pine Script timestamp returns milliseconds? No, in v5 it can be either
    # The user says "time(int unix ts)" so I'll assume seconds
    
    # Convert to datetime
    data['datetime'] = pd.to_datetime(data['time'], unit='s', utc=True)
    
    # London time windows: 8:00-9:45 and 15:00-16:45
    # Convert to London time (Europe/London)
    data['london_time'] = data['datetime'].dt.tz_convert('Europe/London')
    data['hour'] = data['london_time'].dt.hour
    data['minute'] = data['london_time'].dt.minute
    
    # Morning window: 8:00 to 9:45
    # Afternoon window: 15:00 to 16:45
    is_morning_window = (data['hour'] == 8) | ((data['hour'] == 9) & (data['minute'] <= 45))
    is_afternoon_window = (data['hour'] == 15) | ((data['hour'] == 16) & (data['minute'] <= 45))
    in_trading_window = is_morning_window | is_afternoon_window
    
    # Indicators
    # OB conditions
    # isUp(index): close[index] > open[index]
    # isDown(index): close[index] < open[index]
    # isObUp(index): isDown(index + 1) and isUp(index) and close[index] > high[index + 1]
    # isObDown(index): isUp(index + 1) and isDown(index) and close[index] < low[index + 1]
    
    # For obUp = isObUp(1): index=1, so isDown(2) and isUp(1) and close[1] > high[2]
    # For obDown = isObDown(1): index=1, so isUp(2) and isDown(1) and close[1] < low[2]
    
    data['isUp_0'] = data['close'] > data['open']
    data['isDown_0'] = data['close'] < data['open']
    
    # Shift to get previous bars
    data['isUp_1'] = data['isUp_0'].shift(1)
    data['isDown_1'] = data['isDown_0'].shift(1)
    data['isUp_2'] = data['isUp_0'].shift(2)
    data['isDown_2'] = data['isDown_0'].shift(2)
    
    data['close_1'] = data['close'].shift(1)
    data['close_2'] = data['close'].shift(2)
    data['high_1'] = data['high'].shift(1)
    data['high_2'] = data['high'].shift(2)
    data['low_1'] = data['low'].shift(1)
    data['low_2'] = data['low'].shift(2)
    
    # obUp = isDown(2) and isUp(1) and close(1) > high(2)
    obUp = data['isDown_2'] & data['isUp_1'] & (data['close_1'] > data['high_2'])
    # obDown = isUp(2) and isDown(1) and close(1) < low(2)
    obDown = data['isUp_2'] & data['isDown_1'] & (data['close_1'] < data['low_2'])
    
    # FVG conditions
    # bfvg = low > high[2] and volfilt and atrfilt and locfiltb
    # sfvg = high < low[2] and volfilt and atrfilt and locfilts
    
    # Volume filter: volfilt = volume[1] > ta.sma(volume, 9)*1.5
    data['vol_sma9'] = data['volume'].rolling(9).mean()
    data['volfilt'] = data['volume'].shift(1) > data['vol_sma9'] * 1.5
    
    # ATR filter: atr = ta.atr(20) / 1.5, atrfilt = (low - high[2] > atr) or (low[2] - high > atr)
    # Need to implement Wilder ATR
    high_diff = data['high'] - data['low']
    high_low_diff = data['high'] - data['low'].shift(2)
    low_high_diff = data['low'].shift(2) - data['high']
    
    # True range = max(high - low, abs(high - prev close), abs(low - prev close))
    prev_close = data['close'].shift(1)
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - prev_close)
    tr3 = abs(data['low'] - prev_close)
    data['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder ATR(20)
    data['atr'] = data['tr'].ewm(alpha=1/20, adjust=False).mean() / 1.5
    
    atrfilt_bull = data['low'] - data['high'].shift(2) > data['atr']
    atrfilt_bear = data['low'].shift(2) - data['high'] > data['atr']
    data['atrfilt'] = atrfilt_bull | atrfilt_bear
    
    # Trend filter: loc = ta.sma(close, 54), loc2 = loc > loc[1]
    data['loc'] = data['close'].rolling(54).mean()
    data['loc_1'] = data['loc'].shift(1)
    data['loc2'] = data['loc'] > data['loc_1']
    data['locfiltb'] = data['loc2']
    data['locfilts'] = ~data['loc2']
    
    # bfvg and sfvg
    bfvg = (data['low'] > data['high'].shift(2)) & data['volfilt'] & data['atrfilt'] & data['locfiltb']
    sfvg = (data['high'] < data['low'].shift(2)) & data['volfilt'] & data['atrfilt'] & data['locfilts']
    
    # Entry signals
    # Long: bfvg and obUp and in_trading_window
    # Short: sfvg and obDown and in_trading_window
    long_signal = bfvg & obUp & in_trading_window
    short_signal = sfvg & obDown & in_trading_window
    
    # Build entries
    entries = []
    trade_num = 1
    
    # Get indices where signals are true
    long_indices = data[long_signal].index.tolist()
    short_indices = data[short_signal].index.tolist()
    
    # Combine and sort by index to maintain time order
    all_signals = [(idx, 'long') for idx in long_indices] + [(idx, 'short') for idx in short_indices]
    all_signals.sort(key=lambda x: x[0])
    
    for idx, direction in all_signals:
        ts = int(data.loc[idx, 'time'])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = float(data.loc[idx, 'close'])
        
        entries.append({
            'trade_num': trade_num,
            'direction': direction,
            'entry_ts': ts,
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