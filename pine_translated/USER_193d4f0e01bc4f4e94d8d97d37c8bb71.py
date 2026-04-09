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
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('time').sort_index()
    
    # Resample to weekly timeframe
    weekly = df.resample('W').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
    
    # Create shifted versions for previous weeks
    dailyHigh11 = weekly['high']
    dailyLow11 = weekly['low']
    dailyClose11 = weekly['close']
    dailyOpen11 = weekly['open']
    
    dailyHigh21 = dailyHigh11.shift(1)
    dailyLow21 = dailyLow11.shift(1)
    dailyHigh22 = dailyHigh11.shift(2)
    dailyLow22 = dailyLow11.shift(2)
    prevDayHigh11 = dailyHigh11.shift(1)
    prevDayLow11 = dailyLow11.shift(1)
    
    # Calculate indicators
    atr1 = weekly['high'].rolling(14).max() - weekly['low'].rolling(14).min()
    atr1 = atr1.rolling(14).mean()
    
    volfilt11 = df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5
    
    atr211 = (weekly['high'].rolling(20).max() - weekly['low'].rolling(20).min()).rolling(20).mean() / 1.5
    atrfilt11 = ((dailyLow22 - dailyHigh22 > atr211) | (dailyLow22 - dailyHigh11 > atr211))
    
    loc11 = weekly['close'].rolling(54).mean()
    loc211 = loc11 > loc11.shift(1)
    locfiltb11 = loc211
    locfilts11 = ~loc211
    
    # Swing detection
    def swing_detection(h21, h22, h11_shift3, h11_shift4, l21, l22, l11_shift3, l11_shift4):
        is_swing_high = (h21 < h22) & (h11_shift3 < h22) & (h11_shift4 < h22)
        is_swing_low = (l21 > l22) & (l11_shift3 > l22) & (l11_shift4 < l22)
        return is_swing_high, is_swing_low, h22, l22
    
    h11_s3 = dailyHigh11.shift(3)
    h11_s4 = dailyHigh11.shift(4)
    l11_s3 = dailyLow11.shift(3)
    l11_s4 = dailyLow11.shift(4)
    
    is_swing_high11, is_swing_low11, swingHigh11, swingLow11 = swing_detection(
        dailyHigh21, dailyHigh22, h11_s3, h11_s4,
        dailyLow21, dailyLow22, l11_s3, l11_s4
    )
    
    # Track last swing type
    lastSwingType11_series = pd.Series(index=dailyHigh11.index, dtype=str)
    last_swing_high_val = np.nan
    last_swing_low_val = np.nan
    
    for i, idx in enumerate(dailyHigh11.index):
        if is_swing_high11.loc[idx]:
            last_swing_high_val = swingHigh11.loc[idx]
            lastSwingType11_series.loc[idx] = "dailyHigh"
        elif is_swing_low11.loc[idx]:
            last_swing_low_val = swingLow11.loc[idx]
            lastSwingType11_series.loc[idx] = "dailyLow"
        else:
            lastSwingType11_series.loc[idx] = lastSwingType11_series.iloc[i-1] if i > 0 else "none"
    
    # FVG conditions
    bfvg11 = (dailyLow11 > dailyHigh22) & volfilt11.reindex(dailyLow11.index, fill_value=True) & atrfilt11.fillna(True) & locfiltb11.fillna(True)
    sfvg11 = (dailyHigh11 < dailyLow22) & volfilt11.reindex(dailyLow11.index, fill_value=True) & atrfilt11.fillna(True) & locfilts11.fillna(True)
    
    # Entry conditions
    bull_entry = bfvg11 & (lastSwingType11_series == "dailyLow")
    bear_entry = sfvg11 & (lastSwingType11_series == "dailyHigh")
    
    # Merge weekly indicators back to daily data
    df_merged = df.join(weekly[['high', 'low', 'close', 'open']].rename(
        columns={'high': 'weekly_high', 'low': 'weekly_low', 'close': 'weekly_close', 'open': 'weekly_open'}), how='left')
    
    # Forward fill weekly values within each week
    df_merged['weekly_high'] = df_merged['weekly_high'].ffill()
    df_merged['weekly_low'] = df_merged['weekly_low'].ffill()
    df_merged['weekly_close'] = df_merged['weekly_close'].ffill()
    df_merged['weekly_open'] = df_merged['weekly_open'].ffill()
    
    # Map entry signals to daily index
    df_merged['bull_entry'] = False
    df_merged['bear_entry'] = False
    
    for idx in dailyHigh11.index:
        if bull_entry.loc[idx]:
            # Find matching daily bars within this week
            week_start = idx - pd.Timedelta(days=6)
            week_end = idx
            mask = (df_merged.index >= week_start) & (df_merged.index <= week_end)
            df_merged.loc[mask, 'bull_entry'] = True
        if bear_entry.loc[idx]:
            week_start = idx - pd.Timedelta(days=6)
            week_end = idx
            mask = (df_merged.index >= week_start) & (df_merged.index <= week_end)
            df_merged.loc[mask, 'bear_entry'] = True
    
    # Apply London time window filter
    df_merged['hour'] = df_merged.index.hour
    df_merged['minute'] = df_merged.index.minute
    morning_window = ((df_merged['hour'] == 9) & (df_merged['minute'] >= 45)) | \
                     ((df_merged['hour'] >= 10) & (df_merged['hour'] < 12)) | \
                     ((df_merged['hour'] == 12) & (df_merged['minute'] < 45))
    afternoon_window = ((df_merged['hour'] == 14) & (df_merged['minute'] >= 45)) | \
                       ((df_merged['hour'] >= 15) & (df_merged['hour'] < 18)) | \
                       ((df_merged['hour'] == 18) & (df_merged['minute'] < 45))
    in_trading_window = morning_window | afternoon_window
    
    entries = []
    trade_num = 1
    
    for i in range(len(df_merged)):
        row = df_merged.iloc[i]
        ts = int(df_merged.index[i].timestamp())
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        if row['bull_entry'] and in_trading_window.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': row['close'],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': row['close'],
                'raw_price_b': row['close']
            })
            trade_num += 1
        elif row['bear_entry'] and in_trading_window.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': row['close'],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': row['close'],
                'raw_price_b': row['close']
            })
            trade_num += 1
    
    return entries