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
    entries = []
    trade_num = 1
    
    if len(df) < 5:
        return entries
    
    df = df.copy()
    df['ts'] = df['time']
    df['datetime'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df['dayofweek'] = df['datetime'].dt.dayofweek  # 0=Monday, 4=Friday
    
    is_monday = df['dayofweek'] == 0
    is_tuesday = df['dayofweek'] == 1
    is_wednesday = df['dayofweek'] == 2
    is_thursday = df['dayofweek'] == 3
    is_friday = df['dayofweek'] == 4
    
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    
    bull_fvg_bottom = df['high'].shift(3)
    bear_fvg_top = df['low'].shift(3)
    
    df['is_monday'] = is_monday
    df['is_tuesday'] = is_tuesday
    df['is_wednesday'] = is_wednesday
    df['is_thursday'] = is_thursday
    df['is_friday'] = is_friday
    df['ema20'] = ema20
    df['bull_fvg_bottom'] = bull_fvg_bottom
    df['bear_fvg_top'] = bear_fvg_top
    
    is_monday_prev = is_monday.shift(1).fillna(False).astype(bool)
    is_tuesday_prev = is_tuesday.shift(1).fillna(False).astype(bool)
    is_wednesday_prev = is_wednesday.shift(1).fillna(False).astype(bool)
    is_thursday_prev = is_thursday.shift(1).fillna(False).astype(bool)
    is_friday_prev = is_friday.shift(1).fillna(False).astype(bool)
    
    df['new_monday'] = is_monday & ~is_monday_prev
    df['new_tuesday'] = is_tuesday & ~is_tuesday_prev
    df['new_wednesday'] = is_wednesday & ~is_wednesday_prev
    df['new_thursday'] = is_thursday & ~is_thursday_prev
    df['new_friday'] = is_friday & ~is_friday_prev
    
    bullish_bias = df['close'] > ema20
    bearish_bias = df['close'] < ema20
    
    week_high = df['high'].cummax()
    week_low = df['low'].cummin()
    
    bullish_setup = bull_fvg_bottom.notna() & (df['low'] <= bull_fvg_bottom)
    bearish_setup = bear_fvg_top.notna() & (df['high'] >= bear_fvg_top)
    
    wed_low_match = (week_low <= bull_fvg_bottom) & bull_fvg_bottom.notna()
    wed_high_match = (week_high >= bear_fvg_top) & bear_fvg_top.notna()
    thu_low_match = (week_low <= bull_fvg_bottom) & bull_fvg_bottom.notna()
    thu_high_match = (week_high >= bear_fvg_top) & bear_fvg_top.notna()
    
    long_conditions = {
        'monday_bullish': is_monday & bullish_bias,
        'tuesday_classic_low': is_tuesday & bullish_setup,
        'wednesday_low': is_wednesday & wed_low_match,
        'thursday_low': is_thursday & thu_low_match,
        'friday_low': is_friday,
    }
    
    short_conditions = {
        'monday_bearish': is_monday & bearish_bias,
        'tuesday_classic_high': is_tuesday & bearish_setup,
        'wednesday_high': is_wednesday & wed_high_match,
        'thursday_high': is_thursday & thu_high_match,
        'friday_high': is_friday,
    }
    
    triggered_long = pd.Series(False, index=df.index)
    triggered_short = pd.Series(False, index=df.index)
    
    for name, cond in long_conditions.items():
        triggered_long = triggered_long | (cond & ~triggered_long)
    
    for name, cond in short_conditions.items():
        triggered_short = triggered_short | (cond & ~triggered_short)
    
    triggered_long = triggered_long.fillna(False)
    triggered_short = triggered_short.fillna(False)
    
    long_signals = df[triggered_long].index.tolist()
    short_signals = df[triggered_short].index.tolist()
    
    all_signals = [(i, 'long') for i in long_signals] + [(i, 'short') for i in short_signals]
    all_signals.sort(key=lambda x: x[0])
    
    for idx, direction in all_signals:
        ts = int(df['ts'].iloc[idx])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = df['close'].iloc[idx]
        
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
            'raw_price_b': entry_price,
        })
        trade_num += 1
    
    return entries