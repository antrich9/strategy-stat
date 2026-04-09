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
    
    # HTF Bias calculation (Non-Repainting)
    # Resample to HTF (60 min) and get last close of each HTF bar
    htf_period_ms = 60 * 60 * 1000
    df = df.copy()
    df['htf_period'] = (df['time'] // htf_period_ms) * htf_period_ms
    
    htf_df = df.groupby('htf_period').agg({'close': 'last'}).reset_index()
    htf_df.columns = ['htf_period', 'htf_close']
    
    # HTF EMA (50)
    htf_df['htf_ema'] = htf_df['htf_close'].ewm(span=50, adjust=False).mean()
    
    # HTF RSI (14) - Wilder RSI
    delta = htf_df['htf_close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    htf_df['htf_rsi'] = 100 - (100 / (1 + rs))
    
    # Shift by 1 (equivalent to [1] in Pine Script)
    htf_df['htf_close'] = htf_df['htf_close'].shift(1)
    htf_df['htf_ema'] = htf_df['htf_ema'].shift(1)
    htf_df['htf_rsi'] = htf_df['htf_rsi'].shift(1)
    
    # Merge back
    df = df.merge(htf_df[['htf_period', 'htf_close', 'htf_ema', 'htf_rsi']], on='htf_period', how='left')
    df['htf_close'] = df['htf_close'].ffill()
    df['htf_ema'] = df['htf_ema'].ffill()
    df['htf_rsi'] = df['htf_rsi'].ffill()
    
    # HTF Bias conditions
    bull_bias = (df['htf_close'] > df['htf_ema']) & (df['htf_rsi'] > 50)
    bear_bias = (df['htf_close'] < df['htf_ema']) & (df['htf_rsi'] < 50)
    
    # ATR (14) - Wilder ATR
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
    
    # Displacement conditions
    bull_displacement = (df['close'] - df['open']) > df['atr'] * 0.7
    bear_displacement = (df['open'] - df['close']) > df['atr'] * 0.7
    
    # FVG conditions
    bull_fvg = df['low'] > df['high'].shift(2)
    bear_fvg = df['high'] < df['low'].shift(2)
    
    # Fractal calculation
    p = 2
    dh = np.sign(df['high'].diff()).cumsum()
    dl = np.sign(df['low'].diff()).cumsum()
    
    dh_rolling = dh.rolling(p).sum()
    dl_rolling = dl.rolling(p).sum()
    
    bull_fractal = (dh_rolling == -p) & (dh_rolling.shift(p) == p) & (df['high'].shift(p) == df['high'].shift(p).rolling(5).max())
    bear_fractal = (dl_rolling == p) & (dl_rolling.shift(p) == -p) & (df['low'].shift(p) == df['low'].shift(p).rolling(5).min())
    
    # BOS logic with state tracking
    upper_value = np.nan
    lower_value = np.nan
    upper_crossed = False
    lower_crossed = False
    
    bull_bos_arr = pd.Series(False, index=df.index)
    bear_bos_arr = pd.Series(False, index=df.index)
    
    for i in range(5, len(df)):
        if bull_fractal.iloc[i]:
            upper_value = df['high'].iloc[i - p]
            upper_crossed = False
        
        if bear_fractal.iloc[i]:
            lower_value = df['low'].iloc[i - p]
            lower_crossed = False
        
        if not np.isnan(upper_value):
            bull_cross_long = (df['close'].iloc[i] > upper_value) & (df['close'].iloc[i - 1] <= upper_value)
            if bull_cross_long and not upper_crossed:
                bull_bos_arr.iloc[i] = True
                upper_crossed = True
        
        if not np.isnan(lower_value):
            bear_cross_short = (df['close'].iloc[i] < lower_value) & (df['close'].iloc[i - 1] >= lower_value)
            if bear_cross_short and not lower_crossed:
                bear_bos_arr.iloc[i] = True
                lower_crossed = True
    
    # Entry conditions
    bull_entry = bull_bias & bull_bos_arr & bull_displacement & bull_fvg
    bear_entry = bear_bias & bear_bos_arr & bear_displacement & bear_fvg
    
    # Date filter (2023-01-01 to 2099-01-01 in milliseconds)
    start_ts = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime(2099, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    in_date_range = (df['time'] >= start_ts) & (df['time'] <= end_ts)
    
    long_mask = bull_entry & in_date_range
    short_mask = bear_entry & in_date_range
    
    trade_num = 1
    entries = []
    
    for i in range(len(df)):
        if long_mask.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if short_mask.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
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