import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    entries = []
    trade_num = 1
    
    # Calculate Asia session high/low (11 PM - 7 AM UTC)
    asia_high = np.nan
    asia_low = np.nan
    prev_in_session = False
    session_highs = []
    session_lows = []
    in_session_flags = []
    new_session_flags = []
    session_end_flags = []
    
    for i in range(len(df)):
        dt = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc)
        hour = dt.hour
        in_session = (hour >= 23) or (hour < 7)
        
        new_sess = in_session and not prev_in_session
        sess_end = not in_session and prev_in_session
        
        if new_sess:
            asia_high = df['high'].iloc[i]
            asia_low = df['low'].iloc[i]
        elif in_session:
            asia_high = max(asia_high, df['high'].iloc[i])
            asia_low = min(asia_low, df['low'].iloc[i])
        
        session_highs.append(asia_high)
        session_lows.append(asia_low)
        in_session_flags.append(in_session)
        new_session_flags.append(new_sess)
        session_end_flags.append(sess_end)
        prev_in_session = in_session
    
    df['asia_high'] = session_highs
    df['asia_low'] = session_lows
    df['in_session'] = in_session_flags
    df['new_session'] = new_session_flags
    df['session_end'] = session_end_flags
    
    # Valid trade time: 02-05 or 10-12 UTC
    hours = pd.Series([datetime.fromtimestamp(ts, tz=timezone.utc).hour for ts in df['time']], index=df.index)
    df['valid_trade_time'] = ((hours >= 2) & (hours < 5)) | ((hours >= 10) & (hours < 12))
    
    # OB detection functions
    is_up = df['close'] > df['open']
    is_down = df['close'] < df['open']
    
    ob_up = is_down.shift(1) & is_up & (df['close'] > df['high'].shift(1))
    ob_down = is_up.shift(1) & is_down & (df['close'] < df['low'].shift(1))
    
    # FVG detection
    fvg_up = df['low'] > df['high'].shift(2)
    fvg_down = df['high'] < df['low'].shift(2)
    
    # Filters
    vol_filt = df['volume'] > df['volume'].shift(1)
    atr = _wilder_atr(df['high'], df['low'], df['close'], 20)
    atr_val = atr / 1.5
    atr_filt_up = (df['low'] - df['high'].shift(2)) > atr_val
    atr_filt_down = (df['low'].shift(2) - df['high']) > atr_val
    
    loc = df['close'].ewm(span=54, adjust=False).mean()
    loc_trend = loc > loc.shift(1)
    loc_filt_bull = loc_trend
    loc_filt_bear = ~loc_trend
    
    # Bullish FVG
    bull_fvg = fvg_up & vol_filt & atr_filt_up & loc_filt_bull
    # Bearish FVG
    bear_fvg = fvg_down & vol_filt & atr_filt_down & loc_filt_bear
    
    # PDHL Sweep detection
    pdhl_sweep_long = (df['low'] <= df['asia_low'].shift(1)) & (df['close'] > df['asia_low'].shift(1))
    pdhl_sweep_short = (df['high'] >= df['asia_high'].shift(1)) & (df['close'] < df['asia_high'].shift(1))
    
    # Combined entry conditions (bullish: sweep low + bullish FVG/OB)
    long_cond = bull_fvg & pdhl_sweep_long & df['valid_trade_time'] & (~df['in_session'])
    short_cond = bear_fvg & pdhl_sweep_short & df['valid_trade_time'] & (~df['in_session'])
    
    # Generate entries
    for i in range(len(df)):
        if pd.isna(df['asia_high'].iloc[i]) or pd.isna(df['asia_low'].iloc[i]):
            continue
        
        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries

def _wilder_atr(high, low, close, length):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr