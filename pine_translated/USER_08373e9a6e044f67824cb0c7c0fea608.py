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
    
    # London time windows (in local hour)
    df['hour'] = pd.to_datetime(df['time'], unit='ms').dt.hour
    
    is_morning = (df['hour'] >= 8) & (df['hour'] < 10)
    is_afternoon = (df['hour'] >= 14) & (df['hour'] < 17)
    in_trading_window = is_morning | is_afternoon
    
    # FVG detection
    # Bullish FVG: low[2] >= high (gap up)
    # Bearish FVG: low >= high[2] (gap down)
    bull_fvg = df['low'].shift(2) >= df['high']
    bear_fvg = df['low'] >= df['high'].shift(2)
    
    # FVG zone boundaries
    bull_top = np.where(bull_fvg, df['low'].shift(2), np.nan)
    bull_bot = np.where(bull_fvg, df['high'], np.nan)
    bear_top = np.where(bear_fvg, df['high'].shift(2), np.nan)
    bear_bot = np.where(bear_fvg, df['low'], np.nan)
    
    # Active FVG tracking
    bull_fvg_active = bull_fvg.copy()
    bear_fvg_active = bear_fvg.copy()
    
    bull_fvg_top = bull_top.copy()
    bull_fvg_bot = bull_bot.copy()
    bear_fvg_top = bear_top.copy()
    bear_fvg_bot = bear_bot.copy()
    
    # ATR calculation (Wilder)
    tr = np.maximum(df['high'] - df['low'], 
                    np.maximum(np.abs(df['high'] - df['close'].shift(1)),
                               np.abs(df['low'] - df['close'].shift(1))))
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean()
    
    # Swing detection on current timeframe
    swing_high = ((df['high'].shift(1) < df['high'].shift(2)) & 
                  (df['high'].shift(3) < df['high'].shift(2)) & 
                  (df['high'].shift(4) < df['high'].shift(2)))
    swing_low = ((df['low'].shift(1) > df['low'].shift(2)) & 
                 (df['low'].shift(3) > df['low'].shift(2)) & 
                 (df['low'].shift(4) > df['low'].shift(2)))
    
    last_swing_high = df['high'].shift(2).where(swing_high)
    last_swing_high = last_swing_high.ffill()
    last_swing_low = df['low'].shift(2).where(swing_low)
    last_swing_low = last_swing_low.ffill()
    
    # Trend detection
    bullish_count = swing_high.cumsum()
    bearish_count = swing_low.cumsum()
    trend = np.where(bullish_count > bearish_count, 'Bullish', 
                     np.where(bearish_count > bullish_count, 'Bearish', 'Neutral'))
    
    # Entry conditions
    long_entry = np.zeros(len(df), dtype=bool)
    short_entry = np.zeros(len(df), dtype=bool)
    
    for i in range(5, len(df)):
        if not in_trading_window.iloc[i]:
            continue
            
        # Check if price enters bullish FVG zone
        if bull_fvg_active.iloc[i] and not pd.isna(bull_fvg_bot.iloc[i]):
            if df['low'].iloc[i] < bull_fvg_top.iloc[i] and df['low'].iloc[i] > bull_fvg_bot.iloc[i]:
                if trend[i] == 'Bullish':
                    long_entry[i] = True
                    bull_fvg_active.iloc[i] = False
        
        # Check if price enters bearish FVG zone
        if bear_fvg_active.iloc[i] and not pd.isna(bear_fvg_bot.iloc[i]):
            if df['high'].iloc[i] > bear_fvg_bot.iloc[i] and df['high'].iloc[i] < bear_fvg_top.iloc[i]:
                if trend[i] == 'Bearish':
                    short_entry[i] = True
                    bear_fvg_active.iloc[i] = False
        
        # FVG detected at bar - mark as active for future entries
        if bull_fvg.iloc[i]:
            bull_fvg_active.iloc[i] = True
            bull_fvg_top.iloc[i] = bull_top.iloc[i]
            bull_fvg_bot.iloc[i] = bull_bot.iloc[i]
        if bear_fvg.iloc[i]:
            bear_fvg_active.iloc[i] = True
            bear_fvg_top.iloc[i] = bear_top.iloc[i]
            bear_fvg_bot.iloc[i] = bear_bot.iloc[i]
    
    # Generate entries
    for i in range(len(df)):
        if long_entry.iloc[i] if isinstance(long_entry, pd.Series) else long_entry[i]:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
        
        if short_entry.iloc[i] if isinstance(short_entry, pd.Series) else short_entry[i]:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
    
    return entries