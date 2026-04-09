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
    
    results = []
    trade_num = 0
    
    high = df['high']
    low = df['low']
    close = df['close']
    open_col = df['open']
    volume = df['volume']
    time = df['time']
    
    n = len(df)
    
    # Wilder RSI implementation
    def wilder_rsi(src, length):
        delta = src.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilder ATR implementation
    def wilder_atr(high_arr, low_arr, close_arr, length):
        tr1 = high_arr - low_arr
        tr2 = abs(high_arr - close_arr.shift(1))
        tr3 = abs(low_arr - close_arr.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr
    
    # Supertrend implementation
    def supertrend(high_arr, low_arr, close_arr, period, multiplier):
        atr = wilder_atr(high_arr, low_arr, close_arr, period)
        hl2 = (high_arr + low_arr) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr
        supertrend_arr = close_arr.copy()
        direction_arr = pd.Series(0, index=close_arr.index)
        direction_arr.iloc[0] = 1
        in_uptrend = True
        
        for i in range(1, n):
            if close_arr.iloc[i] > upper_band.iloc[i-1]:
                in_uptrend = True
            elif close_arr.iloc[i] < lower_band.iloc[i-1]:
                in_uptrend = False
            else:
                in_uptrend = direction_arr.iloc[i-1] == 1
            
            if in_uptrend:
                direction_arr.iloc[i] = 1
                lower_band.iloc[i] = max(lower_band.iloc[i], lower_band.iloc[i-1]) if i > 0 else lower_band.iloc[i]
            else:
                direction_arr.iloc[i] = -1
                upper_band.iloc[i] = min(upper_band.iloc[i], upper_band.iloc[i-1]) if i > 0 else upper_band.iloc[i]
        
        return supertrend_arr, direction_arr
    
    # Detect new day
    ts = pd.to_datetime(df['time'], unit='s', utc=True)
    dates = ts.dt.date
    new_day = dates != dates.shift(1).fillna(dates.iloc[0])
    new_day.iloc[0] = True
    
    # Previous day high/low
    pd_high = pd.Series(np.nan, index=df.index)
    pd_low = pd.Series(np.nan, index=df.index)
    temp_high = high.iloc[0] if not np.isnan(high.iloc[0]) else -np.inf
    temp_low = low.iloc[0] if not np.isnan(low.iloc[0]) else np.inf
    
    for i in range(n):
        if new_day.iloc[i]:
            pd_high.iloc[i] = temp_high if temp_high != -np.inf else np.nan
            pd_low.iloc[i] = temp_low if temp_low != np.inf else np.nan
            temp_high = high.iloc[i]
            temp_low = low.iloc[i]
        else:
            temp_high = max(temp_high, high.iloc[i]) if not np.isnan(high.iloc[i]) else temp_high
            temp_low = min(temp_low, low.iloc[i]) if not np.isnan(low.iloc[i]) else temp_low
            pd_high.iloc[i] = pd_high.iloc[i-1] if i > 0 else np.nan
            pd_low.iloc[i] = pd_low.iloc[i-1] if i > 0 else np.nan
    
    # ATR calculations
    atr_14 = wilder_atr(high, low, close, 14)
    atr_20 = wilder_atr(high, low, close, 20)
    atr2 = atr_20 / 1.5
    
    # Volume filter
    vol_sma = volume.rolling(9).mean()
    vol_filt = volume.shift(1) > vol_sma * 1.5
    
    # ATR filter
    atr_filt = (low - high.shift(2) > atr2) | (low.shift(2) - high > atr2)
    
    # Trend filter (SMA 54)
    sma_54 = close.rolling(54).mean()
    loc211 = sma_54 > sma_54.shift(1)
    
    # Swing detection
    daily_high_11 = high
    daily_low_11 = low
    daily_high_21 = high.shift(1)
    daily_low_21 = low.shift(1)
    daily_high_22 = high.shift(2)
    daily_low_22 = low.shift(2)
    
    is_swing_high = (daily_high_21 < daily_high_22) & (daily_high_11.shift(3) < daily_high_22) & (daily_high_11.shift(4) < daily_high_22)
    is_swing_low = (daily_low_21 > daily_low_22) & (daily_low_11.shift(3) > daily_low_22) & (daily_low_11.shift(4) > daily_low_22)
    
    # FVG detection
    bfvg = (low > high.shift(2)) & loc211 & atr_filt
    sfvg = (high < low.shift(2)) & (~loc211) & atr_filt
    
    # Supertrend
    supertrend_vals, supertrend_dir = supertrend(high, low, close, 10, 3)
    is_supertrend_bullish = supertrend_dir == 1
    is_supertrend_bearish = supertrend_dir == -1
    
    # Time window detection (simplified - using hour extraction)
    ts_local = ts.dt.tz_convert('Europe/London')
    hour = ts_local.dt.hour
    minute = ts_local.dt.minute
    time_minutes = hour * 60 + minute
    
    morning_start = 6 * 60 + 45
    morning_end = 9 * 60 + 45
    afternoon_start = 14 * 60 + 45
    afternoon_end = 16 * 60 + 45
    
    in_morning = (time_minutes >= morning_start) & (time_minutes < morning_end)
    in_afternoon = (time_minutes >= afternoon_start) & (time_minutes < afternoon_end)
    in_trading_window = in_morning | in_afternoon
    
    # Track last swing type
    last_swing_type = pd.Series('none', index=df.index)
    last_swing_high = pd.Series(np.nan, index=df.index)
    last_swing_low = pd.Series(np.nan, index=df.index)
    
    current_swing_type = 'none'
    current_swing_high = np.nan
    current_swing_low = np.nan
    
    for i in range(n):
        if is_swing_high.iloc[i] and not np.isnan(daily_high_22.iloc[i]):
            current_swing_high = daily_high_22.iloc[i]
            current_swing_type = 'dailyHigh'
        if is_swing_low.iloc[i] and not np.isnan(daily_low_22.iloc[i]):
            current_swing_low = daily_low_22.iloc[i]
            current_swing_type = 'dailyLow'
        
        last_swing_type.iloc[i] = current_swing_type
        last_swing_high.iloc[i] = current_swing_high
        last_swing_low.iloc[i] = current_swing_low
    
    # Entry conditions
    long_condition = bfvg & (last_swing_type == 'dailyLow') & is_supertrend_bullish & in_trading_window
    short_condition = sfvg & (last_swing_type == 'dailyHigh') & is_supertrend_bearish & in_trading_window
    
    # Skip first few bars where indicators may be NaN
    start_idx = max(54, 20)  # based on SMA(54) and ATR(20)
    
    for i in range(start_idx, n):
        if pd.isna(atr_14.iloc[i]) or pd.isna(sma_54.iloc[i]):
            continue
        
        entry_price = close.iloc[i]
        
        if long_condition.iloc[i]:
            trade_num += 1
            ts_entry = int(time.iloc[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts_entry,
                'entry_time': datetime.fromtimestamp(ts_entry, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
        
        if short_condition.iloc[i]:
            trade_num += 1
            ts_entry = int(time.iloc[i])
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts_entry,
                'entry_time': datetime.fromtimestamp(ts_entry, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
    
    return results