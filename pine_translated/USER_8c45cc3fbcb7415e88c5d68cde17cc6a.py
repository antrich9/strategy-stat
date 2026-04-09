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
    trade_num = 1
    
    # Extract series
    high = df['high']
    low = df['low']
    close = df['close']
    time = df['time']
    
    # Wilder ATR calculation (period 14)
    def wilder_atr(period):
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Wilder smoothing
        atr = tr.ewm(alpha=1.0/period, adjust=False).mean()
        return atr
    
    atr = wilder_atr(14)
    atr1 = wilder_atr(14)
    atrMultiplier = 3.0
    
    # SuperTrend calculation for current timeframe (simulating daily and 4h aligned)
    def calc_supertrend(hl, atr_val, factor):
        hl2 = hl.copy()
        atr_val2 = atr_val.copy()
        
        upperband = hl2 + factor * atr_val2
        lowerband = hl2 - factor * atr_val2
        
        supertrend = np.full(len(hl2), np.nan)
        direction = np.full(len(hl2), 0)  # 1 = downtrend, -1 = uptrend
        
        for i in range(1, len(hl2)):
            if not np.isnan(supertrend[i-1]):
                if hl2[i] < supertrend[i-1]:
                    supertrend[i] = upperband[i]
                    direction[i] = 1
                else:
                    supertrend[i] = lowerband[i]
                    direction[i] = -1
            else:
                supertrend[i] = upperband[i]
                direction[i] = 1
            
            if direction[i] == 1:
                if not np.isnan(supertrend[i-1]) and supertrend[i] < supertrend[i-1]:
                    supertrend[i] = supertrend[i-1]
            else:
                if not np.isnan(supertrend[i-1]) and supertrend[i] > supertrend[i-1]:
                    supertrend[i] = supertrend[i-1]
        
        return supertrend, direction
    
    st_daily, dir_daily = calc_supertrend(close, atr, 3.0)
    st_4h, dir_4h = calc_supertrend(close, atr, 3.0)
    
    dir_daily_series = pd.Series(dir_daily, index=df.index)
    dir_4h_series = pd.Series(dir_4h, index=df.index)
    
    # SuperTrend Bullish/Bearish (both aligned)
    bullish = (dir_daily_series < 0) & (dir_4h_series < 0)
    bearish = (dir_daily_series > 0) & (dir_4h_series > 0)
    
    # Previous Day High/Low calculation
    df_temp = df.copy()
    df_temp['datetime'] = pd.to_datetime(df_temp['time'], unit='s', utc=True)
    df_temp['day'] = df_temp['datetime'].dt.date
    
    pdHigh_arr = np.full(len(df), np.nan)
    pdLow_arr = np.full(len(df), np.nan)
    tempHigh_arr = np.full(len(df), np.nan)
    tempLow_arr = np.full(len(df), np.nan)
    
    for i in range(len(df)):
        if i == 0:
            tempHigh_arr[i] = high.iloc[i]
            tempLow_arr[i] = low.iloc[i]
        else:
            if df_temp['day'].iloc[i] != df_temp['day'].iloc[i-1]:
                pdHigh_arr[i] = tempHigh_arr[i-1]
                pdLow_arr[i] = tempLow_arr[i-1]
                tempHigh_arr[i] = high.iloc[i]
                tempLow_arr[i] = low.iloc[i]
            else:
                tempHigh_arr[i] = max(tempHigh_arr[i-1] if not np.isnan(tempHigh_arr[i-1]) else -np.inf, high.iloc[i])
                tempLow_arr[i] = min(tempLow_arr[i-1] if not np.isnan(tempLow_arr[i-1]) else np.inf, low.iloc[i])
    
    pdHigh = pd.Series(pdHigh_arr, index=df.index)
    pdLow = pd.Series(pdLow_arr, index=df.index)
    
    # FVG conditions
    lookback_bars = 12
    
    bear_fvg1 = (high < low.shift(2)) & (close.shift(1) < low.shift(2))
    bull_fvg1 = (low > high.shift(2)) & (close.shift(1) > high.shift(2))
    
    def barssince(series):
        result = np.full(len(df), -1)
        count = -1
        for i in range(len(df)):
            if series.iloc[i]:
                count = 0
            elif count >= 0:
                count += 1
            result[i] = count if count >= 0 else -1
        return pd.Series(result, index=df.index)
    
    bull_since = barssince(bull_fvg1)
    bear_since = barssince(bear_fvg1)
    
    bull_cond_1 = bull_fvg1 & (bull_since <= lookback_bars)
    bear_cond_1 = bear_fvg1 & (bear_since <= lookback_bars)
    
    combined_low_bull = np.where(bull_cond_1, 
                                np.maximum(high.shift(bull_since.astype(int)), high.shift(2)), 
                                np.nan)
    combined_high_bull = np.where(bull_cond_1, 
                                   np.minimum(low.shift((bull_since + 2).astype(int)), low), 
                                   np.nan)
    bull_range = combined_high_bull - combined_low_bull
    bull_result = bull_cond_1 & (bull_range >= 0)
    
    combined_low_bear = np.where(bear_cond_1, 
                                  np.maximum(high.shift((bear_since + 2).astype(int)), high), 
                                  np.nan)
    combined_high_bear = np.where(bear_cond_1, 
                                  np.minimum(low.shift(bear_since.astype(int)), low.shift(2)), 
                                  np.nan)
    bear_range = combined_high_bear - combined_low_bear
    bear_result = bear_cond_1 & (bear_range >= 0)
    
    # London Session Windows
    df_temp['hour'] = df_temp['datetime'].dt.hour
    df_temp['minute'] = df_temp['datetime'].dt.minute
    
    isWithinWindow1 = ((df_temp['hour'] == 7) & (df_temp['minute'] >= 45)) | \
                      ((df_temp['hour'] >= 8) & (df_temp['hour'] < 11)) | \
                      ((df_temp['hour'] == 11) & (df_temp['minute'] < 45))
    isWithinWindow2 = ((df_temp['hour'] >= 13) & (df_temp['hour'] < 16)) | \
                      ((df_temp['hour'] == 16) & (df_temp['minute'] < 45))
    in_trading_window = isWithinWindow1 | isWithinWindow2
    
    # Long entry condition
    long_entry_cond = bull_result & in_trading_window & bullish
    # Short entry condition
    short_entry_cond = bear_result & in_trading_window & bearish
    
    for i in range(len(df)):
        if pd.isna(atr.iloc[i]) or pd.isna(atr1.iloc[i]):
            continue
        
        entry_ts = int(time.iloc[i])
        entry_price = float(close.iloc[i])
        
        if long_entry_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_entry_cond.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries