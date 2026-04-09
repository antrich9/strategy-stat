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
    
    # Input parameters from Pine Script
    hull_length = 21
    hull_source = 'close'
    adx_length = 14
    adx_threshold = 25.0
    bbp_length = 13
    exp_fast = 20
    exp_slow = 40
    exp_sensitivity = 150
    exp_tr_length = 100
    exp_deadzone_mult = 3.7
    use_adaptive_deadzone = True
    exp_deadzone = 20.0
    stop_length = 22
    stop_mult = 3.0
    entry_filter = 3
    
    # Parse source
    if hull_source == 'close':
        src = df['close'].copy()
    elif hull_source == 'open':
        src = df['open'].copy()
    elif hull_source == 'high':
        src = df['high'].copy()
    elif hull_source == 'low':
        src = df['low'].copy()
    elif hull_source == 'hl2':
        src = (df['high'] + df['low']) / 2
    elif hull_source == 'hlc3':
        src = (df['high'] + df['low'] + df['close']) / 3
    elif hull_source == 'ohlc4':
        src = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    else:
        src = df['close'].copy()
    
    close = df['close'].copy()
    high = df['high'].copy()
    low = df['low'].copy()
    
    # Adaptive Hull Moving Average calculation
    def hull_calc(series, length):
        half_length = int(length / 2)
        sqrt_length = int(np.round(np.sqrt(length)))
        
        wma1 = series.ewm(span=half_length, adjust=False).mean()
        wma2 = series.ewm(span=length, adjust=False).mean()
        diff = 2 * wma1 - wma2
        hull = diff.ewm(span=sqrt_length, adjust=False).mean()
        return hull
    
    atp_hull = hull_calc(src, hull_length)
    atp_hull_prev = atp_hull.shift(1)
    atp_hull_slope = atp_hull - atp_hull_prev
    
    # ADX calculation (Wilder smoothing)
    high_prev = high.shift(1)
    low_prev = low.shift(1)
    close_prev = close.shift(1)
    
    tr1 = high - low
    tr2 = np.abs(high - close_prev)
    tr3 = np.abs(low - close_prev)
    atp_tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    dm_plus_raw = high - high_prev
    dm_minus_raw = low_prev - low
    dm_plus = np.where((dm_plus_raw > dm_minus_raw) & (dm_plus_raw > 0), dm_plus_raw, 0.0)
    dm_minus = np.where((dm_minus_raw > dm_plus_raw) & (dm_minus_raw > 0), dm_minus_raw, 0.0)
    
    # Wilder smoothed TR
    atp_tr_smooth = pd.Series(index=df.index, dtype=float)
    atp_tr_smooth.iloc[0] = atp_tr.iloc[0]
    for i in range(1, len(df)):
        atp_tr_smooth.iloc[i] = atp_tr_smooth.iloc[i-1] - (atp_tr_smooth.iloc[i-1] / adx_length) + atp_tr.iloc[i]
    
    # Wilder smoothed DM+
    atp_dm_plus_smooth = pd.Series(index=df.index, dtype=float)
    atp_dm_plus_smooth.iloc[0] = dm_plus[0]
    for i in range(1, len(df)):
        atp_dm_plus_smooth.iloc[i] = atp_dm_plus_smooth.iloc[i-1] - (atp_dm_plus_smooth.iloc[i-1] / adx_length) + dm_plus[i]
    
    # Wilder smoothed DM-
    atp_dm_minus_smooth = pd.Series(index=df.index, dtype=float)
    atp_dm_minus_smooth.iloc[0] = dm_minus[0]
    for i in range(1, len(df)):
        atp_dm_minus_smooth.iloc[i] = atp_dm_minus_smooth.iloc[i-1] - (atp_dm_minus_smooth.iloc[i-1] / adx_length) + dm_minus[i]
    
    atp_di_plus = (atp_dm_plus_smooth / atp_tr_smooth) * 100
    atp_di_minus = (atp_dm_minus_smooth / atp_tr_smooth) * 100
    
    dx = np.abs(atp_di_plus - atp_di_minus) / (atp_di_plus + atp_di_minus) * 100
    atp_adx = dx.rolling(adx_length).mean()
    
    atp_adx_strong = atp_adx > adx_threshold
    
    # Explosion Indicator (Waddah Attar style)
    atp_fast_ma = close.ewm(span=exp_fast, adjust=False).mean()
    atp_slow_ma = close.ewm(span=exp_slow, adjust=False).mean()
    atp_macd = atp_fast_ma - atp_slow_ma
    atp_macd_prev = atp_macd.shift(1)
    atp_macd_diff = (atp_macd - atp_macd_prev) * exp_sensitivity
    
    atp_trend_up = np.where(atp_macd_diff >= 0, atp_macd_diff, 0.0)
    atp_trend_down = np.where(atp_macd_diff < 0, -atp_macd_diff, 0.0)
    
    atp_exp_up = pd.Series(atp_trend_up, index=df.index)
    atp_exp_down = pd.Series(atp_trend_down, index=df.index)
    
    # Adaptive deadzone
    atp_tr_rma = atp_tr.ewm(span=exp_tr_length, adjust=False).mean()
    atp_adaptive_deadzone = atp_tr_rma * exp_deadzone_mult
    atp_current_deadzone = atp_adaptive_deadzone if use_adaptive_deadzone else exp_deadzone
    
    atp_exp_bull_signal = atp_exp_up > atp_current_deadzone
    atp_exp_bear_signal = atp_exp_down > atp_current_deadzone
    
    # Bull Bear Power
    atp_ema = close.ewm(span=bbp_length, adjust=False).mean()
    atp_bull_power = high - atp_ema
    atp_bear_power = low - atp_ema
    atp_bbp = atp_bull_power + atp_bear_power
    atp_bbp_prev = atp_bbp.shift(1)
    
    atp_bbp_bull_cross = (atp_bbp > 0) & (atp_bbp_prev <= 0)
    atp_bbp_bear_cross = (atp_bbp < 0) & (atp_bbp_prev >= 0)
    
    atp_bbp_bull = atp_bbp > 0
    atp_bbp_bear = atp_bbp < 0
    
    # Trend bias
    atp_trend_bias = np.where(atp_hull_slope > 0, 1, np.where(atp_hull_slope < 0, -1, 0))
    atp_trend_bias = pd.Series(atp_trend_bias, index=df.index)
    
    # Entry conditions
    long_condition = atp_exp_bull_signal & atp_adx_strong & (atp_bbp_bull | atp_bbp_bull_cross) & (atp_trend_bias >= 0)
    short_condition = atp_exp_bear_signal & atp_adx_strong & (atp_bbp_bear | atp_bbp_bear_cross) & (atp_trend_bias <= 0)
    
    # Build entries
    entries = []
    trade_num = 1
    last_entry_idx = -entry_filter
    
    for i in range(1, len(df)):
        if pd.isna(atp_hull.iloc[i]) or pd.isna(atp_adx.iloc[i]) or pd.isna(atp_bbp.iloc[i]):
            continue
        
        direction = None
        if long_condition.iloc[i] and (i - last_entry_idx) > entry_filter:
            direction = 'long'
            last_entry_idx = i
        elif short_condition.iloc[i] and (i - last_entry_idx) > entry_filter:
            direction = 'short'
            last_entry_idx = i
        
        if direction:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
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