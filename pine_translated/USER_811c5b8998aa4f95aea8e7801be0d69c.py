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
    
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # OB conditions
    is_up = close > open_
    is_down = close < open_
    
    # Bullish OB: down[1] and up[0] and close[0] > high[1]
    is_ob_up = (is_down.shift(1)) & (is_up) & (close > high.shift(1))
    
    # Bearish OB: up[1] and down[0] and close[0] < low[1]
    is_ob_down = (is_up.shift(1)) & (is_down) & (close < low.shift(1))
    
    # FVG conditions
    # Bullish FVG: low[0] > high[2]
    is_fvg_up = low > high.shift(2)
    
    # Bearish FVG: high[0] < low[2]
    is_fvg_down = high < low.shift(2)
    
    # ATR filter (Wilder)
    atr = ta_atr(high, low, close, 20)
    atr_threshold = atr / 1.5
    
    # Volume filter
    vol_sma = volume.rolling(9).mean()
    vol_filt = volume.shift(1) > vol_sma * 1.5
    
    # ATR filter
    atr_filt_long = (low - high.shift(2)) > atr_threshold
    atr_filt_short = (high.shift(2) - low) > atr_threshold
    
    # Trend filter
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    loc_filt_long = loc2
    loc_filt_short = ~loc2
    
    # Stacked OB+FVG conditions
    stacked_bullish = is_fvg_up & is_ob_up
    stacked_bearish = is_fvg_down & is_ob_down
    
    # Combined entry conditions (with filters)
    # Using inp defaults: volfilt=true, atrfilt=true, locfilt=true
    bull_cond = stacked_bullish & vol_filt & atr_filt_long & loc_filt_long
    bear_cond = stacked_bearish & vol_filt & atr_filt_short & loc_filt_short
    
    # Trading window logic
    ts = df['time']
    dt = pd.to_datetime(ts, unit='s', utc=True)
    hour = dt.dt.hour
    minute = dt.dt.minute
    month = dt.dt.month
    dayofweek = dt.dt.dayofweek  # 0=Monday, 6=Sunday
    
    # DST check for UK
    # DST starts last Sunday in March, ends last Sunday in October
    is_dst = (
        ((month > 3) | ((month == 3) & ((dayofweek == 6) & (dt.dt.day >= 25) | (dayofweek < 6)))) &
        ((month < 10) | ((month == 10) & ((dayofweek == 6) & (dt.dt.day < 25) | (dayofweek < 6))))
    )
    
    # Trading windows: 07:00-10:59 and 15:00-16:59
    window1_start = 7
    window1_end = 10
    window1_end_min = 59
    window2_start = 15
    window2_end = 16
    window2_end_min = 59
    
    in_window1 = (hour >= window1_start) & (hour <= window1_end)
    in_window1 &= ~((hour == window1_end) & (minute > window1_end_min))
    
    in_window2 = (hour >= window2_start) & (hour <= window2_end)
    in_window2 &= ~((hour == window2_end) & (minute > window2_end_min))
    
    in_trading_window = in_window1 | in_window2
    
    # Final entry conditions
    long_entries = bull_cond & in_trading_window
    short_entries = bear_cond & in_trading_window
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(close.iloc[i]):
            continue
        
        if long_entries.iloc[i]:
            ts_val = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts_val,
                'entry_time': datetime.fromtimestamp(ts_val, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        elif short_entries.iloc[i]:
            ts_val = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts_val,
                'entry_time': datetime.fromtimestamp(ts_val, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    return entries

def ta_atr(high, low, close, length=14):
    """Wilder's ATR calculation"""
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr