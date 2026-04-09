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
    
    # Convert time to datetime with UTC timezone
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert('Europe/London')
    
    # Time window conditions
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    
    # Morning window: 08:00 to 09:45
    isWithinMorningWindow = (
        (df['hour'] == 8) | 
        ((df['hour'] == 9) & (df['minute'] <= 45))
    )
    
    # Afternoon window: 15:00 to 16:45
    isWithinAfternoonWindow = (
        ((df['hour'] == 15) & (df['minute'] >= 0)) | 
        ((df['hour'] == 16) & (df['minute'] <= 45))
    )
    
    df['in_trading_window'] = isWithinMorningWindow | isWithinAfternoonWindow
    
    # FVG conditions - bullish: low > high[2]
    df['bfvg_raw'] = df['low'] > df['high'].shift(2)
    
    # FVG conditions - bearish: high < low[2]
    df['sfvg_raw'] = df['high'] < df['low'].shift(2)
    
    # Volume filter: volume[1] > sma(volume, 9) * 1.5 (inp1 default false)
    volume_sma9 = df['volume'].rolling(9).mean()
    df['volfilt'] = df['volume'].shift(1) > volume_sma9 * 1.5
    
    # ATR filter (inp2 default false) - using Wilder ATR
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    df['atrfilt'] = (df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr)
    
    # Trend filter (inp3 default false)
    loc = close.rolling(54).mean()
    loc_prev = loc.shift(1)
    loc2 = loc > loc_prev
    df['locfiltb'] = loc2
    df['locfilts'] = ~loc2
    
    # Final FVG conditions with filters
    df['bfvg'] = df['bfvg_raw'] & df['volfilt'] & df['atrfilt'] & df['locfiltb']
    df['sfvg'] = df['sfvg_raw'] & df['volfilt'] & df['atrfilt'] & df['locfilts']
    
    # Entry conditions
    df['bullish_entry'] = df['bfvg'] & df['in_trading_window']
    df['bearish_entry'] = df['sfvg'] & df['in_trading_window']
    
    # Generate entries
    entries = []
    trade_num = 1
    
    # Bullish entries
    for i in range(len(df)):
        if df['bullish_entry'].iloc[i] and not pd.isna(df['close'].iloc[i]):
            entry_ts = int(df['time'].iloc[i])
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
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
    
    # Bearish entries
    for i in range(len(df)):
        if df['bearish_entry'].iloc[i] and not pd.isna(df['close'].iloc[i]):
            entry_ts = int(df['time'].iloc[i])
            entry_price = df['close'].iloc[i]
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
    
    return entries