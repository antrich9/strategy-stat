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

    n = len(df)
    if n < 5:
        return entries

    time_arr = df['time'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    close_arr = df['close'].values
    volume_arr = df['volume'].values

    # Weekly data (using weekly security in Pine, approximated by weekly resample in Python)
    weekly_df = df.set_index('time').resample('W')['high'].max().reset_index()
    weekly_df.columns = ['time', 'weekly_high']
    weekly_low_df = df.set_index('time').resample('W')['low'].min().reset_index()
    weekly_low_df.columns = ['time', 'weekly_low']
    weekly_close_df = df.set_index('time').resample('W')['close'].last().reset_index()
    weekly_close_df.columns = ['time', 'weekly_close']
    weekly_open_df = df.set_index('time').resample('W')['open'].first().reset_index()
    weekly_open_df.columns = ['time', 'weekly_open']

    weekly_df = weekly_df.merge(weekly_low_df, on='time').merge(weekly_close_df, on='time').merge(weekly_open_df, on='time')
    weekly_df = weekly_df.sort_values('time').reset_index(drop=True)

    # Shift weekly data to align with daily bars
    weekly_df['weekly_high_shifted'] = weekly_df['weekly_high'].shift(1)
    weekly_df['weekly_low_shifted'] = weekly_df['weekly_low'].shift(1)
    weekly_df['weekly_high_shifted2'] = weekly_df['weekly_high'].shift(2)
    weekly_df['weekly_low_shifted2'] = weekly_df['weekly_low'].shift(2)

    # Merge weekly data back to daily dataframe
    df_merged = df.merge(weekly_df[['time', 'weekly_high', 'weekly_low', 'weekly_close', 'weekly_open', 
                                     'weekly_high_shifted', 'weekly_low_shifted', 
                                     'weekly_high_shifted2', 'weekly_low_shifted2']], on='time', how='left')

    # Fill NaN values with previous valid values
    df_merged = df_merged.ffill()

    # ATR calculation (Wilder)
    tr1 = df_merged['high'] - df_merged['low']
    tr2 = np.abs(df_merged['high'] - df_merged['close'].shift(1))
    tr3 = np.abs(df_merged['low'] - df_merged['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    atr20 = tr.ewm(alpha=1/20, adjust=False).mean()

    # Volume filter
    sma9_vol = df_merged['volume'].rolling(9).mean()
    volfilt = sma9_vol * 1.5

    # ATR filter
    atr211 = atr20 / 1.5

    # Trend filter (SMA 54)
    sma54 = df_merged['close'].rolling(54).mean()
    sma54_prev = sma54.shift(1)
    loc21 = sma54 > sma54_prev

    # Swing detection using weekly data
    dailyHigh21 = df_merged['weekly_high_shifted'].values
    dailyHigh22 = df_merged['weekly_high_shifted2'].values
    dailyLow21 = df_merged['weekly_low_shifted'].values
    dailyLow22 = df_merged['weekly_low_shifted2'].values

    # Also need dailyHigh11 (current week high) and dailyLow11 (current week low)
    dailyHigh11 = df_merged['weekly_high'].values
    dailyLow11 = df_merged['weekly_low'].values

    # Swing detection
    is_swing_high = np.zeros(n, dtype=bool)
    is_swing_low = np.zeros(n, dtype=bool)

    for i in range(4, n):
        if not np.isnan(dailyHigh21[i]) and not np.isnan(dailyHigh22[i]) and not np.isnan(dailyLow21[i]) and not np.isnan(dailyLow22[i]):
            # For swing high: dailyHigh21 < dailyHigh22 and dailyHigh11[3] < dailyHigh22 and dailyHigh11[4] < dailyHigh22
            if dailyHigh21[i] < dailyHigh22[i] and dailyHigh11[max(0, i-3)] < dailyHigh22[i] and dailyHigh11[max(0, i-4)] < dailyHigh22[i]:
                is_swing_high[i] = True
            # For swing low: dailyLow21 > dailyLow22 and dailyLow11[3] > dailyLow22 and dailyLow11[4] > dailyLow22
            if dailyLow21[i] > dailyLow22[i] and dailyLow11[max(0, i-3)] > dailyLow22[i] and dailyLow11[max(0, i-4)] > dailyLow22[i]:
                is_swing_low[i] = True

    # Track last swing type
    lastSwingType = [None] * n
    for i in range(n):
        if is_swing_high[i]:
            lastSwingType[i] = "dailyHigh"
        elif is_swing_low[i]:
            lastSwingType[i] = "dailyLow"
        elif i > 0:
            lastSwingType[i] = lastSwingType[i-1]

    # FVG conditions
    bfvg = np.zeros(n, dtype=bool)
    sfvg = np.zeros(n, dtype=bool)

    for i in range(n):
        if np.isnan(dailyLow11[i]) or np.isnan(dailyHigh22[i]) or np.isnan(dailyHigh11[i]) or np.isnan(dailyLow22[i]):
            continue
        
        cond_vol = volume_arr[max(0, i-1)] > volfilt.iloc[max(0, i-1)] if not np.isnan(volfilt.iloc[max(0, i-1)]) else True
        cond_atr = (dailyLow22[i] - dailyHigh22[i] > atr211.iloc[i]) or (dailyLow22[i] - dailyHigh11[i] > atr211.iloc[i])
        cond_trend = loc21.iloc[i]
        
        if dailyLow11[i] > dailyHigh22[i] and cond_vol and cond_atr and cond_trend:
            bfvg[i] = True
        
        if dailyHigh11[i] < dailyLow22[i] and cond_vol and cond_atr and not cond_trend:
            sfvg[i] = True

    # Entry signals
    bullish_fvg_entry = np.zeros(n, dtype=bool)
    bearish_fvg_entry = np.zeros(n, dtype=bool)

    for i in range(n):
        if bfvg[i] and lastSwingType[i] == "dailyLow":
            bullish_fvg_entry[i] = True
        if sfvg[i] and lastSwingType[i] == "dailyHigh":
            bearish_fvg_entry[i] = True

    # Build entries list
    for i in range(n):
        if bullish_fvg_entry[i]:
            ts = int(df_merged['time'].iloc[i])
            entry_price = float(df_merged['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if bearish_fvg_entry[i]:
            ts = int(df_merged['time'].iloc[i])
            entry_price = float(df_merged['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries