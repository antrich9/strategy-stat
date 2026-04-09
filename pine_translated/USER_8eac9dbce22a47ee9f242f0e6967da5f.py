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
    
    high = df['high']
    low = df['low']
    close = df['close']
    open_col = df['open']
    volume = df['volume']
    
    # FVG detection (fair value gaps)
    bull_fvg = (low > high.shift(2)) & (close.shift(1) > high.shift(2))
    bear_fvg = (high < low.shift(2)) & (close.shift(1) < low.shift(2))
    
    # Volume filter
    volfilt = volume.shift(1) > volume.rolling(9).mean() * 1.5
    
    # ATR filter
    def wilder_atr(high, low, close, length=20):
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        alpha = 1.0 / length
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        return atr
    
    atr_series = wilder_atr(high, low, close, 20) / 1.5
    atrfilt = ((low - high.shift(2) > atr_series) | (low.shift(2) - high > atr_series))
    
    # Trend filter
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # Combined FVG conditions with filters
    bull_fvg_combined = bull_fvg & volfilt & atrfilt & locfiltb
    bear_fvg_combined = bear_fvg & volfilt & atrfilt & locfilts
    
    # Breakaway imbalances (top)
    top_imbalance_bway = (low.shift(2) <= open_col.shift(1)) & (high >= close.shift(1)) & (close < low.shift(1))
    top_imb_xbway = (low.shift(2) <= open_col.shift(1)) & (high >= close.shift(1)) & (close > low.shift(1))
    
    # Breakaway imbalances (bottom)
    bottom_imbalance_bway = (high.shift(2) >= open_col.shift(1)) & (low <= close.shift(1)) & (close > high.shift(1))
    bottom_imb_xbway = (high.shift(2) >= open_col.shift(1)) & (low <= close.shift(1)) & (close < high.shift(1))
    
    # BPR settings (using default values)
    lookback_bars = 12
    threshold = 0.0
    
    # BPR bullish detection
    bull_since = bull_fvg_combined.shift(1).rolling(window=lookback_bars+1).apply(lambda x: x.sum(), raw=True)
    bull_cond_1 = bull_fvg_combined & (bull_since <= lookback_bars)
    
    # BPR bearish detection  
    bear_since = bear_fvg_combined.shift(1).rolling(window=lookback_bars+1).apply(lambda x: x.sum(), raw=True)
    bear_cond_1 = bear_fvg_combined & (bear_since <= lookback_bars)
    
    # Entry conditions: bull_fvg signals for long, bear_fvg signals for short
    long_entry = bull_fvg_combined
    short_entry = bear_fvg_combined
    
    # Skip bars with NaN in key indicators
    valid_idx = ~(high.shift(2).isna() | low.shift(2).isna() | close.shift(1).isna() | volume.shift(1).isna())
    long_entry = long_entry & valid_idx
    short_entry = short_entry & valid_idx
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        direction = None
        entry_price = df['close'].iloc[i]
        
        if long_entry.iloc[i]:
            direction = 'long'
        elif short_entry.iloc[i]:
            direction = 'short'
        
        if direction:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
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