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
    
    # Detect new day
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['day'] = df['datetime'].dt.date
    df['new_day'] = df['day'].diff().fillna(True)
    
    # Calculate Previous Day High and Low
    pdHigh = np.nan
    pdLow = np.nan
    tempHigh = np.nan
    tempLow = np.nan
    
    pdHigh_arr = np.zeros(len(df))
    pdLow_arr = np.zeros(len(df))
    
    for i in range(len(df)):
        if df['new_day'].iloc[i]:
            pdHigh = tempHigh
            pdLow = tempLow
            tempHigh = df['high'].iloc[i]
            tempLow = df['low'].iloc[i]
        else:
            tempHigh = df['high'].iloc[i] if np.isnan(tempHigh) else np.maximum(tempHigh, df['high'].iloc[i])
            tempLow = df['low'].iloc[i] if np.isnan(tempLow) else np.minimum(tempLow, df['low'].iloc[i])
        pdHigh_arr[i] = pdHigh if not np.isnan(pdHigh) else 0
        pdLow_arr[i] = pdLow if not np.isnan(pdLow) else 0
    
    df['pdHigh'] = pdHigh_arr
    df['pdLow'] = pdLow_arr
    
    # Sweep detection
    sweptHigh = False
    sweptLow = False
    sweptHigh_arr = np.zeros(len(df), dtype=bool)
    sweptLow_arr = np.zeros(len(df), dtype=bool)
    
    for i in range(len(df)):
        if df['new_day'].iloc[i]:
            sweptHigh = False
            sweptLow = False
        sweepHighNow = not sweptHigh and df['high'].iloc[i] > df['pdHigh'].iloc[i]
        sweepLowNow = not sweptLow and df['low'].iloc[i] < df['pdLow'].iloc[i]
        if sweepHighNow:
            sweptHigh = True
        if sweepLowNow:
            sweptLow = True
        sweptHigh_arr[i] = sweptHigh
        sweptLow_arr[i] = sweptLow
    
    # ATR calculation (Wilder)
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift(1))
    tr3 = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr1 = tr.rolling(window=14, min_periods=14).mean()
    for i in range(14, len(atr1)):
        if not np.isnan(atr1.iloc[i]):
            atr1.iloc[i] = (atr1.iloc[i-1] * 13 + tr.iloc[i]) / 14
    df['atr1'] = atr1
    
    atr2 = tr.rolling(window=20, min_periods=20).mean()
    for i in range(20, len(atr2)):
        if not np.isnan(atr2.iloc[i]):
            atr2.iloc[i] = (atr2.iloc[i-1] * 19 + tr.iloc[i]) / 20
    df['atr2'] = atr2 / 1.5
    
    # Volume filter (disabled by default)
    vol_filt = df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5
    vol_filt = vol_filt.fillna(True)
    
    # ATR filter (disabled by default)
    atr_filt = ((df['low'] - df['high'].shift(2) > df['atr2']) | (df['low'].shift(2) - df['high'] > df['atr2']))
    atr_filt = atr_filt.fillna(True)
    
    # Trend filter (disabled by default)
    loc11 = df['close'].rolling(54).mean()
    loc211 = loc11 > loc11.shift(1)
    loc211 = loc211.fillna(False)
    loc_filt_bull = loc211.fillna(True)
    loc_filt_short = (~loc211).fillna(True)
    
    # Daily data references
    dailyHigh11 = df['high']
    dailyLow11 = df['low']
    dailyClose11 = df['close']
    dailyOpen11 = df['open']
    
    dailyHigh21 = df['high'].shift(1)
    dailyLow21 = df['low'].shift(1)
    dailyHigh22 = df['high'].shift(2)
    dailyLow22 = df['low'].shift(2)
    
    # Swing detection
    is_swing_high = (dailyHigh21 < dailyHigh22) & (dailyHigh11.shift(3) < dailyHigh22) & (dailyHigh11.shift(4) < dailyHigh22)
    is_swing_low = (dailyLow21 > dailyLow22) & (dailyLow11.shift(3) > dailyLow22) & (dailyLow11.shift(4) > dailyLow22)
    
    last_swing_high = np.nan
    last_swing_low = np.nan
    lastSwingType = "none"
    
    lastSwingType_arr = []
    for i in range(len(df)):
        if is_swing_high.iloc[i]:
            last_swing_high = dailyHigh22.iloc[i] if not np.isnan(dailyHigh22.iloc[i]) else last_swing_high
            lastSwingType = "dailyHigh"
        if is_swing_low.iloc[i]:
            last_swing_low = dailyLow22.iloc[i] if not np.isnan(dailyLow22.iloc[i]) else last_swing_low
            lastSwingType = "dailyLow"
        lastSwingType_arr.append(lastSwingType)
    
    df['lastSwingType'] = lastSwingType_arr
    
    # London time window
    london_offset = df['datetime'].dt.utcoffset().fillna(pd.Timedelta(seconds=0)).dt.total_seconds().astype(int)
    df['local_hour'] = (df['datetime'].dt.hour + london_offset / 3600) % 24
    df['local_minute'] = df['datetime'].dt.minute
    
    morning_start_h, morning_start_m = 6, 45
    morning_end_h, morning_end_m = 9, 45
    afternoon_start_h, afternoon_start_m = 14, 45
    afternoon_end_h, afternoon_end_m = 16, 45
    
    def in_time_window(row, sh, sm, eh, em):
        current_minutes = row['local_hour'] * 60 + row['local_minute']
        start_minutes = sh * 60 + sm
        end_minutes = eh * 60 + em
        return start_minutes <= current_minutes < end_minutes
    
    in_morning = df.apply(lambda r: in_time_window(r, morning_start_h, morning_start_m, morning_end_h, morning_end_m), axis=1)
    in_afternoon = df.apply(lambda r: in_time_window(r, afternoon_start_h, afternoon_start_m, afternoon_end_h, afternoon_end_m), axis=1)
    in_trading_window = in_morning | in_afternoon
    
    # Supertrend
    super_period = 10
    super_mult = 3
    
    hl2 = (df['high'] + df['low']) / 2
    atr_st = df['atr1'].copy()
    
    upper_band = hl2 + super_mult * atr_st
    lower_band = hl2 - super_mult * atr_st
    
    super_trend = np.zeros(len(df))
    super_direction = np.zeros(len(df))
    
    for i in range(len(df)):
        if i == 0:
            super_trend[i] = lower_band.iloc[i]
            super_direction[i] = 1
        else:
            if close_prev > super_trend[i-1]:
                super_direction[i] = 1
            elif close_prev < super_trend[i-1]:
                super_direction[i] = -1
            else:
                super_direction[i] = super_direction[i-1]
            
            if super_direction[i] == 1:
                super_trend[i] = max(lower_band.iloc[i], super_trend[i-1])
            else:
                super_trend[i] = min(upper_band.iloc[i], super_trend[i-1])
        close_prev = df['close'].iloc[i]
    
    df['super_direction'] = super_direction
    is_super_bull = df['super_direction'] == 1
    is_super_bear = df['super_direction'] == -1
    
    # FVG detection
    bull_fvg = (df['low'] > df['high'].shift(2)) & vol_filt & atr_filt & loc_filt_bull
    bear_fvg = (df['high'] < df['low'].shift(2)) & vol_filt & atr_filt & loc_filt_short
    
    bull_fvg = bull_fvg.fillna(False)
    bear_fvg = bear_fvg.fillna(False)
    
    # Entry conditions
    is_bullish_leg = (bull_fvg) & (df['lastSwingType'] == "dailyLow")
    is_bearish_leg = (bear_fvg) & (df['lastSwingType'] == "dailyHigh")
    
    long_cond = is_bullish_leg & in_trading_window & is_super_bull
    short_cond = is_bearish_leg & in_trading_window & is_super_bear
    
    long_cond = long_cond.fillna(False)
    short_cond = short_cond.fillna(False)
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif short_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries