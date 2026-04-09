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
    
    df = df.copy()
    
    # Time window extraction (London time - simplified)
    df['hour'] = pd.to_datetime(df['time'], unit='s').dt.hour
    df['minute'] = pd.to_datetime(df['time'], unit='s').dt.minute
    
    # London morning (8:00-9:55) and afternoon (14:00-16:55)
    is_morning = ((df['hour'] == 8) & (df['minute'] >= 0)) | \
                 ((df['hour'] == 9) & (df['minute'] <= 55))
    is_afternoon = ((df['hour'] == 14) & (df['minute'] >= 0)) | \
                   ((df['hour'] == 16) & (df['minute'] <= 55))
    df['in_trading_window'] = is_morning | is_afternoon
    
    # ADX (Wilder smoothing)
    adxLength = 14
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    
    dm_plus = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    dm_minus = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
    
    # Wilder ATR
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                               abs(df['low'] - df['close'].shift(1))))
    
    atr_smooth = tr.ewm(alpha=1/adxLength, adjust=False).mean()
    di_plus_smooth = (dm_plus.ewm(alpha=1/adxLength, adjust=False).mean() / atr_smooth) * 100
    di_minus_smooth = (dm_minus.ewm(alpha=1/adxLength, adjust=False).mean() / atr_smooth) * 100
    
    dx = (abs(di_plus_smooth - di_minus_smooth) / (di_plus_smooth + di_minus_smooth)) * 100
    adx = dx.ewm(alpha=1/adxLength, adjust=False).mean()
    
    df['strong_trend'] = adx > 25
    df['adx'] = adx
    
    # EMA Trend Direction
    df['ema_fast'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=50, adjust=False).mean()
    df['trend_bullish'] = df['ema_fast'] > df['ema_slow']
    df['trend_bearish'] = df['ema_fast'] < df['ema_slow']
    
    # FVG Detection (Fair Value Gap)
    # Bullish FVG: low >= high[2] (current low above high 2 bars ago)
    # Bearish FVG: high <= low[2] (current high below low 2 bars ago)
    df['fvg_top'] = np.where(df['low'] >= df['high'].shift(2), df['low'], np.nan)
    df['fvg_bottom'] = np.where(df['high'] <= df['low'].shift(2), df['high'], np.nan)
    
    # For bullish FVG entries: previous bar low < FVG top and current bar close > FVG top
    df['bullish_fvg_active'] = df['fvg_top'].notna()
    df['bull_fvg_touched'] = (df['low'].shift(1) < df['fvg_top'].shift(1)) & df['bullish_fvg_active'].shift(1)
    df['bull_entry'] = (df['close'] > df['fvg_top'].shift(1)) & df['bull_fvg_touched']
    
    # For bearish FVG entries: previous bar high > FVG bottom and current bar close < FVG bottom
    df['bearish_fvg_active'] = df['fvg_bottom'].notna()
    df['bear_fvg_touched'] = (df['high'].shift(1) > df['fvg_bottom'].shift(1)) & df['bearish_fvg_active'].shift(1)
    df['bear_entry'] = (df['close'] < df['fvg_bottom'].shift(1)) & df['bear_fvg_touched']
    
    # Combined entry conditions
    df['long_entry'] = (df['in_trading_window'] & 
                        df['strong_trend'] & 
                        df['trend_bullish'] & 
                        df['bull_entry'])
    
    df['short_entry'] = (df['in_trading_window'] & 
                         df['strong_trend'] & 
                         df['trend_bearish'] & 
                         df['bear_entry'])
    
    # Generate entries
    entries = []
    trade_num = 1
    
    # Long entries
    long_bars = df[df['long_entry']].iterrows()
    for i, row in long_bars:
        entry_price = row['close']
        entries.append({
            'trade_num': trade_num,
            'direction': 'long',
            'entry_ts': int(row['time']),
            'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
            'entry_price_guess': float(entry_price),
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': float(entry_price),
            'raw_price_b': float(entry_price)
        })
        trade_num += 1
    
    # Short entries
    short_bars = df[df['short_entry']].iterrows()
    for i, row in short_bars:
        entry_price = row['close']
        entries.append({
            'trade_num': trade_num,
            'direction': 'short',
            'entry_ts': int(row['time']),
            'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
            'entry_price_guess': float(entry_price),
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': float(entry_price),
            'raw_price_b': float(entry_price)
        })
        trade_num += 1
    
    return entries