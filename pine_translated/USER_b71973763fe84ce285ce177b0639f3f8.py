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
    
    if len(df) < 3:
        return []
    
    # Create a datetime column for resampling
    df_copy = df.copy()
    df_copy['dt'] = pd.to_datetime(df_copy['time'], unit='s', utc=True)
    df_copy.set_index('dt', inplace=True)
    
    # Resample to 4H
    h4_data = df_copy.resample('240T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    h4_high = h4_data['high']
    h4_low = h4_data['low']
    h4_close = h4_data['close']
    h4_volume = h4_data['volume']
    
    # 4H ATR (Wilder smoothing via ewm)
    tr = np.maximum(h4_high - h4_low, 
                    np.maximum(np.abs(h4_high - h4_close.shift(1)),
                               np.abs(h4_low - h4_close.shift(1))))
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    
    # 4H Volume SMA
    vol_sma = h4_volume.rolling(9).mean()
    
    # 4H Trend SMA
    trend_sma = h4_close.rolling(54).mean()
    
    # 4H FVG conditions
    bull_fvg = h4_low > h4_high.shift(2)
    bear_fvg = h4_high < h4_low.shift(2)
    
    vol_filt = h4_volume.shift(1) > vol_sma * 1.5
    atr_filt = atr.shift(1) > 0
    trend_bull = trend_sma > trend_sma.shift(1)
    trend_bear = trend_sma < trend_sma.shift(1)
    
    bull_fvg_final = bull_fvg & vol_filt & atr_filt & trend_bull
    bear_fvg_final = bear_fvg & vol_filt & atr_filt & trend_bear
    
    last_fvg = 0
    trade_num = 0
    entries = []
    
    for i in range(len(df)):
        ts = int(df['time'].iloc[i])
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        time_mins = hour * 60 + minute
        in_window = (420 <= time_mins < 705) or (840 <= time_mins < 885)
        
        if not in_window:
            continue
        
        # Find corresponding 4H bar index
        dt_utc = dt.replace(tzinfo=timezone.utc)
        h4_idx_candidates = h4_data.index[h4_data.index <= dt_utc]
        if len(h4_idx_candidates) == 0:
            continue
        h4_idx = h4_idx_candidates[-1]
        h4_pos = h4_data.index.get_loc(h4_idx)
        
        if h4_pos < 2 or pd.isna(h4_high.iloc[h4_pos]) or pd.isna(h4_low.iloc[h4_pos]):
            continue
        
        direction = None
        if bull_fvg_final.iloc[h4_pos] and last_fvg == -1:
            direction = 'long'
        elif bear_fvg_final.iloc[h4_pos] and last_fvg == 1:
            direction = 'short'
        
        if direction:
            trade_num += 1
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': dt.isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            last_fvg = 1 if direction == 'long' else -1
        elif bull_fvg_final.iloc[h4_pos]:
            last_fvg = 1
        elif bear_fvg_final.iloc[h4_pos]:
            last_fvg = -1
    
    return entries