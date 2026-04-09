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
    results = []
    trade_num = 1
    
    # ATR Filter (ATR / 1.5)
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift(1))
    tr3 = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(20).mean()
    atr_filt = atr / 1.5
    
    # Volume Filter
    vol_sma = df['volume'].rolling(9).mean()
    vol_filt = df['volume'].shift(1) > vol_sma * 1.5
    
    # Trend Filter (SMA 54)
    loc = df['close'].rolling(54).mean()
    loc_trend_up = loc > loc.shift(1)
    loc_trend_down = loc < loc.shift(1)
    
    # Bullish FVG: low > high[2] with filters
    bfvg = (df['low'] > df['high'].shift(2)) & vol_filt & (tr < atr_filt) & loc_trend_up
    
    # Bearish FVG: high < low[2] with filters
    sfvg = (df['high'] < df['low'].shift(2)) & vol_filt & (tr < atr_filt) & loc_trend_down
    
    # OB Conditions (using shifted access equivalent)
    is_up_curr = df['close'] > df['open']
    is_down_curr = df['close'] < df['open']
    is_up_prev = df['open'].shift(1) < df['close'].shift(1)
    is_down_prev = df['open'].shift(1) > df['close'].shift(1)
    
    ob_up = is_down_prev & is_up_curr & (df['close'] > df['high'].shift(1))
    ob_down = is_up_prev & is_down_curr & (df['close'] < df['low'].shift(1))
    
    # FVG conditions
    fvg_up = df['low'] > df['high'].shift(2)
    fvg_down = df['high'] < df['low'].shift(2)
    
    # Stacked OB + FVG conditions
    bull_stacked = ob_up.shift(1) & fvg_up
    bear_stacked = ob_down.shift(1) & fvg_down
    
    # Time window filtering (London time zones)
    ts = pd.to_datetime(df['time'], unit='s', utc=True)
    london_morning_start = ts.dt.tz_localize(None).dt.tz_localize('Europe/London').dt.normalize() + pd.Timedelta(hours=7, minutes=45)
    london_morning_end = ts.dt.tz_localize(None).dt.tz_localize('Europe/London').dt.normalize() + pd.Timedelta(hours=9, minutes=45)
    london_afternoon_start = ts.dt.tz_localize(None).dt.tz_localize('Europe/London').dt.normalize() + pd.Timedelta(hours=14, minutes=45)
    london_afternoon_end = ts.dt.tz_localize(None).dt.tz_localize('Europe/London').dt.normalize() + pd.Timedelta(hours=16, minutes=45)
    
    in_london_morning = (ts >= london_morning_start) & (ts < london_morning_end)
    in_london_afternoon = (ts >= london_afternoon_start) & (ts < london_afternoon_end)
    in_trading_window = in_london_morning | in_london_afternoon
    
    # Combined entry conditions
    long_condition = bull_stacked & in_trading_window
    short_condition = bear_stacked & in_trading_window
    
    # Iterate through bars to generate entries
    for i in range(1, len(df)):
        if np.isnan(atr.iloc[i]):
            continue
        
        entry_price = df['close'].iloc[i]
        ts_val = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts_val, tz=timezone.utc).isoformat()
        
        if long_condition.iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts_val,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts_val,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return results