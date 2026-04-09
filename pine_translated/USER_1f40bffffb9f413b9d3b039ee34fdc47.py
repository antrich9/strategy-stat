import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    
    # Ensure datetime index or column exists
    if 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df['datetime'] = df.index
    else:
        df['datetime'] = df.index
    
    # Previous day high/low (shift by 1 bar - approximating previous day)
    df['prevDayHigh'] = df['high'].shift(1)
    df['prevDayLow'] = df['low'].shift(1)
    
    # Daily swing detection (using forward-looking in Pine, approximating with rolling)
    # Pine: dailyHigh2 < dailyHigh22 and dailyHigh[3] < dailyHigh22 and dailyHigh[4] < dailyHigh22
    # Simulating with current bar comparing to next bars (approx - use accessible data)
    df['dailyHigh22'] = df['high'].shift(-2)
    df['dailyHigh2'] = df['high'].shift(-1)
    df['dailyLow22'] = df['low'].shift(-2)
    df['dailyLow2'] = df['low'].shift(-1)
    
    is_swing_high = (df['dailyHigh2'] < df['dailyHigh22']) & (df['high'].shift(-3) < df['dailyHigh22']) & (df['high'].shift(-4) < df['dailyHigh22'])
    is_swing_low = (df['dailyLow2'] > df['dailyLow22']) & (df['low'].shift(-3) > df['dailyLow22']) & (df['low'].shift(-4) > df['dailyLow22'])
    
    # 4H FVG detection using shift to approximate 4H candle structure
    # Pine: low_4h > high_4h_2 (bullish), high_4h < low_4h_2 (bearish)
    # Approximating 4H candles with available data shifts
    low_4h = df['low'].shift(2)
    high_4h_2 = df['high'].shift(4)
    high_4h = df['high'].shift(2)
    low_4h_2 = df['low'].shift(4)
    
    bfvg_condition = (low_4h > high_4h_2) & df['close'].shift(1).notna()
    sfvg_condition = (high_4h < low_4h_2) & df['close'].shift(1).notna()
    
    # Track last swing type (state variable)
    lastSwingType = pd.Series("none", index=df.index)
    for i in range(1, len(df)):
        if is_swing_high.iloc[i]:
            lastSwingType.iloc[i] = "dailyHigh"
        elif is_swing_low.iloc[i]:
            lastSwingType.iloc[i] = "dailyLow"
        else:
            lastSwingType.iloc[i] = lastSwingType.iloc[i-1]
    
    # PDH/PDL sweep detection
    sweepHighNow = (df['high'] > df['prevDayHigh']) & df['prevDayHigh'].notna()
    sweepLowNow = (df['low'] < df['prevDayLow']) & df['prevDayLow'].notna()
    
    # London session time windows
    hours = df['datetime'].dt.hour
    minutes = df['datetime'].dt.minute
    time_minutes = hours * 60 + minutes
    
    in_trading_window = ((time_minutes >= 9*60) & (time_minutes < 12*60) | 
                          (time_minutes >= 14*60) & (time_minutes < 14*60 + 45))
    
    # Entry conditions
    bull_entry = bfvg_condition & (lastSwingType == "dailyLow") & sweepLowNow & in_trading_window
    bear_entry = sfvg_condition & (lastSwingType == "dailyHigh") & sweepHighNow & in_trading_window
    
    traded = False
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(df['close'].iloc[i]):
            continue
        
        if bull_entry.iloc[i] and not traded:
            traded = True
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        
        elif bear_entry.iloc[i] and not traded:
            traded = True
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        
        elif traded:
            traded = False
    
    return entries