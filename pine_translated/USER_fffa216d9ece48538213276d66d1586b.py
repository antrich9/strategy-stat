import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
    # Calculate EMA50 on chart timeframe
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Multi-timeframe analysis - resample to higher timeframes
    # Daily timeframe
    daily_ohlc = df['close'].resample('D').ohlc()
    daily_ema = daily_ohlc['close'].ewm(span=50, adjust=False).mean()
    daily_rejection = daily_ohlc['close'] < daily_ema
    df['daily_rejection'] = df.index.map(lambda x: daily_rejection.get(x.normalize(), False))
    
    # Hourly timeframe
    hourly_agg = df[['low', 'high']].resample('H').agg({'low': 'min', 'high': 'max'})
    hourly_lhh = (hourly_agg['low'] < hourly_agg['low'].shift(1)) & (hourly_agg['high'] < hourly_agg['high'].shift(1))
    df['hourly_lower_high_low'] = df.index.map(lambda x: hourly_lhh.get(x.floor('H'), False))
    
    # 4H timeframe
    fourhour_ohlc = df['close'].resample('4H').ohlc()
    fourhour_ema = fourhour_ohlc['close'].ewm(span=50, adjust=False).mean()
    fourhour_rejection = fourhour_ohlc['close'] < fourhour_ema
    df['fourhour_rejection'] = df.index.map(lambda x: fourhour_rejection.get(x.floor('4H'), False))
    
    # breakAndRetest function on 5-minute chart
    bearish_break = (df['low'].shift(1) < df['low'].shift(2)) & (df['close'].shift(1) < df['low'].shift(2))
    bearish_retest = (df['high'] > df['low'].shift(2)) & (df['close'] < df['low'].shift(2))
    break_and_retest = bearish_break & bearish_retest
    
    # Entry condition for main strategy
    entry_condition = df['daily_rejection'] & df['hourly_lower_high_low'] & df['fourhour_rejection'] & break_and_retest
    
    # London session logic
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    is_london_session = (df['hour'] >= 8) & (df['hour'] < 16)
    
    # Identify London open high/low bars
    london_open_mask = is_london_session & (df['hour'] == 8) & (df['minute'] == 0)
    
    # Initialize london session variables
    london_open_high = pd.Series(np.nan, index=df.index)
    london_open_low = pd.Series(np.nan, index=df.index)
    
    # Set values at the start of London session
    for idx in df[london_open_mask].index:
        london_open_high.loc[idx:] = df.loc[idx, 'high']
        london_open_low.loc[idx:] = df.loc[idx, 'low']
        break  # Only take the first 8:00 bar of the session
    
    # Forward fill within session
    london_open_high = london_open_high.ffill()
    london_open_low = london_open_low.ffill()
    
    # London breakout conditions
    london_breakout_long = is_london_session & (df['close'] > london_open_high) & (df['close'].shift(1) <= london_open_high)
    london_breakout_short = is_london_session & (df['close'] < london_open_low) & (df['close'].shift(1) >= london_open_low)
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        # Skip if not enough history for indicators
        if i < 2:
            continue
        
        # Main strategy short entry
        if entry_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        
        # London session entries (need at least 1 bar history)
        if i >= 1:
            if london_breakout_long.iloc[i]:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                })
                trade_num += 1
            
            if london_breakout_short.iloc[i]:
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                })
                trade_num += 1
    
    return entries