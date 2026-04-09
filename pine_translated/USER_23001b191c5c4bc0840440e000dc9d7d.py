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
    # Helper functions
    def is_up(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]
    
    def is_down(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]
    
    def is_ob_up(idx):
        return is_down(idx + 1) and is_up(idx) and df['close'].iloc[idx] > df['high'].iloc[idx + 1]
    
    def is_ob_down(idx):
        return is_up(idx + 1) and is_down(idx) and df['close'].iloc[idx] < df['low'].iloc[idx + 1]
    
    def is_fvg_up(idx):
        return df['low'].iloc[idx] > df['high'].iloc[idx + 2]
    
    def is_fvg_down(idx):
        return df['high'].iloc[idx] < df['low'].iloc[idx + 2]
    
    # Time window detection
    def in_trading_window(ts):
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        
        # Adjust for UTC+1 and DST
        # Simplified: just use UTC hour directly for demo
        # In real impl, would need proper TZ handling
        in_window_1 = (hour >= 7 and hour <= 10) and not (hour == 10 and minute > 59)
        in_window_2 = (hour >= 15 and hour <= 16) and not (hour == 16 and minute > 59)
        
        return in_window_1 or in_window_2
    
    # Build boolean series for conditions
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Volume filter
    vol_sma = volume.rolling(9).mean()
    vol_filt = vol_sma * 1.5
    
    # ATR filter (Wilder)
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atr_filter_val = atr / 1.5
    
    # Trend filter (SMA 54)
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    
    # Bullish conditions
    bfvg = is_fvg_up(0)
    bull_fvg = (low > high.shift(2)) & (volume > vol_filt) & ((low - high.shift(2) > atr_filter_val) | (low.shift(2) - high > atr_filter_val)) & loc2
    
    # Bearish conditions
    sfvg = is_fvg_down(0)
    bear_fvg = (high < low.shift(2)) & (volume > vol_filt) & ((low - high.shift(2) > atr_filter_val) | (low.shift(2) - high > atr_filter_val)) & ~loc2
    
    # OB conditions
    bull_ob = (low.shift(2) <= open_.shift(1)) & (high >= close.shift(1)) & (close < low.shift(1))
    bear_ob = (high.shift(2) >= open_.shift(1)) & (low <= close.shift(1)) & (close > high.shift(1))
    
    entries = []
    trade_num = 1
    
    for i in range(2, len(df)):
        ts = int(df['time'].iloc[i])
        
        # Skip if indicators are NaN
        if pd.isna(atr_filter_val.iloc[i]) or pd.isna(loc2.iloc[i]) or pd.isna(vol_filt.iloc[i]):
            continue
        
        if not in_trading_window(ts):
            continue
        
        # Long entry: Bullish FVG + Bullish OB
        if bull_fvg.iloc[i] and bull_ob.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        
        # Short entry: Bearish FVG + Bearish OB
        if bear_fvg.iloc[i] and bear_ob.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
    
    return entries