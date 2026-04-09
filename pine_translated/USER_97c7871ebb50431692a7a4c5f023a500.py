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
    trade_num = 0
    
    num_bars = len(df)
    if num_bars == 0:
        return entries
    
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # ATR (Wilder) for stop loss
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # ATR for filter (ta.atr(20) / 1.5)
    atr20 = tr.ewm(alpha=1/20, adjust=False).mean()
    atr_filter = atr20 / 1.5
    
    # Daily close (for higherTimeframeTrend) - use current bar as current daily candle
    daily_close = close.copy()
    daily_ema50 = daily_close.ewm(span=50, adjust=False).mean()
    
    # State variables
    last_swing_high = np.nan
    last_swing_low = np.nan
    last_fvg = 0
    
    # Process bars
    for i in range(num_bars):
        if i < 6:
            continue
        
        # Swing detection
        main_bar_high = high.iloc[i-5]
        main_bar_low = low.iloc[i-5]
        
        is_swing_high = high.iloc[i-4] < main_bar_high and high.iloc[i-6] < main_bar_high
        is_swing_low = low.iloc[i-4] > main_bar_low and low.iloc[i-6] > main_bar_low
        
        if is_swing_high:
            last_swing_high = main_bar_high
        if is_swing_low:
            last_swing_low = main_bar_low
        
        # FVG detection
        bfvg = low.iloc[i] > high.iloc[i-2]
        sfvg = high.iloc[i] < low.iloc[i-2]
        
        # Update lastFVG based on FVG
        if bfvg:
            last_fvg = 1
        elif sfvg:
            last_fvg = -1
        
        # Entry conditions
        bullish_entry = False
        bearish_entry = False
        
        daily_ema_val = daily_ema50.iloc[i]
        if not np.isnan(daily_ema_val):
            if bfvg and last_fvg == -1 and daily_close.iloc[i] > daily_ema_val:
                bullish_entry = True
            if sfvg and last_fvg == 1 and daily_close.iloc[i] < daily_ema_val:
                bearish_entry = True
        
        # Execute entries (strategy.position_size == 0 equivalent - no open position)
        if bullish_entry:
            trade_num += 1
            ts = df['time'].iloc[i]
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
        elif bearish_entry:
            trade_num += 1
            ts = df['time'].iloc[i]
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
    
    return entries