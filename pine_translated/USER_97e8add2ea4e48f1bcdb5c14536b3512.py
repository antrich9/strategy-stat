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
    
    # Convert time to datetime in Europe/London timezone
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Europe/London')
    
    # 1. Resample to 4H for higher timeframe FVG detection
    df_indexed = df.set_index('datetime')
    htf_4h = df_indexed[['high', 'low']].resample('240T').agg({'high': 'max', 'low': 'min'})
    htf_4h = htf_4h.dropna()
    
    # 2. Calculate 4H FVG conditions
    htf_4h['htf_bull_fvg'] = htf_4h['low'].shift(2) > htf_4h['high']
    htf_4h['htf_bear_fvg'] = htf_4h['high'].shift(2) < htf_4h['low']
    
    # 3. Track last FVG direction
    htf_4h['last_htf_fvg'] = np.where(htf_4h['htf_bull_fvg'], 'bullish', 
                                       np.where(htf_4h['htf_bear_fvg'], 'bearish', np.nan))
    htf_4h['last_htf_fvg'] = htf_4h['last_htf_fvg'].ffill()
    
    # 4. Forward fill 4H data to 15m bars
    htf_aligned = htf_4h.reindex(df_indexed.index, method='ffill')
    
    # 5. Calculate 15m FVG conditions
    df['bull_fvg_15m'] = df['low'].shift(2) > df['high']
    df['bear_fvg_15m'] = df['high'].shift(2) < df['low']
    
    # 6. Trading windows (Europe/London)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    h = df['hour']
    m = df['minute']
    is_window1 = ((h == 7) & (m >= 45)) | (h == 8) | ((h == 9) & (m <= 45))
    is_window2 = ((h == 14) & (m >= 45)) | (h == 15) | ((h == 16) & (m <= 45))
    df['in_trading_window'] = is_window1 | is_window2
    
    # 7. Entry conditions
    df['last_htf_fvg'] = htf_aligned['last_htf_fvg'].values
    df['enter_short'] = (df['last_htf_fvg'] == 'bullish') & df['bear_fvg_15m']
    df['enter_long'] = (df['last_htf_fvg'] == 'bearish') & df['bull_fvg_15m']
    
    # 8. Generate entries
    entries = []
    trade_num = 1