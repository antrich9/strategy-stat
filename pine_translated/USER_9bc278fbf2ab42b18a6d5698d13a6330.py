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

    df = df.copy()
    df['ts'] = df['time']
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    df['total_minutes'] = df['hour'] * 60 + df['minute']

    fvgWaitBars = 10
    fvgMinTicks = 3

    usePDH = True
    usePDL = True
    usePWH = True
    usePWL = True
    sweepTicks = 5

    df_indexed = df.set_index('ts')

    try:
        daily = df_indexed.resample('D').agg({'high': 'max', 'low': 'min', 'open': 'first', 'close': 'last'})
        daily = daily.dropna(subset=['high', 'low'])
        daily['prev_day_high'] = daily['high'].shift(1)
        daily['prev_day_low'] = daily['low'].shift(1)
        daily_aligned = daily.reindex(df_indexed.index)
        pdh = daily_aligned['prev_day_high']
        pdl = daily_aligned['prev_day_low']
    except:
        pdh = df['high'] * np.nan
        pdl = df['low'] * np.nan

    try:
        weekly = df_indexed.resample('W').agg({'high': 'max', 'low': 'min', 'open': 'first', 'close': 'last'})
        weekly = weekly.dropna(subset=['high', 'low'])
        weekly['prev_week_high'] = weekly['high'].shift(1)
        weekly['prev_week_low'] = weekly['low'].shift(1)
        weekly_aligned = weekly.reindex(df_indexed.index)
        pwh = weekly_aligned['prev_week_high']
        pwl = weekly_aligned['prev_week_low']
    except:
        pwh = df['high'] * np.nan
        pwl = df['low'] * np.nan

    tick_size = (df['high'] / df['high'].replace(0, 1)).clip(upper=0.1).replace(0, 0.01).iloc[0] if len(df) > 0 else 0.01
    tick_size = min(max(tick_size, 0.0001), 1.0)
    sweepBuffer = sweepTicks * tick_size

    sellsideSweepPDL = usePDL & pd.notna(pdl) & (df['low'] < (pdl - sweepBuffer)) & (df['close'] > pdl)
    sellsideSweepPWL = usePWL & pd.notna(pwl) & (df['low'] < (pwl - sweepBuffer)) & (df['close'] > pwl)
    sellsideSweep = sellsideSweepPDL | sellsideSweepPWL

    buysideSweepPDH = usePDH & pd.notna(pdh) & (df['high'] > (pdh + sweepBuffer)) & (df['close'] < pdh)
    buysideSweepPWH = usePWH & pd.notna(pwh) & (df['high'] > (pwh + sweepBuffer)) & (df['close'] < pwh)
    buysideSweep = buysideSweepPDH | buysideSweepPWH

    high_2 = df['high'].shift(2)
    low_2 = df['low'].shift(2)

    bullishFVG = df['low'] > high_2
    bullishFVGSize = np.where(bullishFVG, df['low'] - high_2, 0)
    bullishFVGValid = bullishFVG & (bullishFVGSize / tick_size >= fvgMinTicks)

    bearishFVG = df['high'] < low_2
    bearishFVGSize = np.where(bearishFVG, low_2 - df['high'], 0)
    bearishFVGValid = bearishFVG & (bearishFVGSize / tick_size >= fvgMinTicks)

    high_1 = df['high'].shift(1)
    low_1 = df['low'].shift(1)
    close_1 = df['close'].shift(1)

    bear_fvg1 = (df['high'] < low_2) & (close_1 < low_2)
    bull_fvg1 = (df['low'] > high_2) & (close_1 > high_2)

    lookback_bars = 12
    threshold = 0.0

    bull_since = np.full(len(df), np.nan)
    for i in range(len(df)):
        for j in range(1, lookback_bars + 1):
            if i - j >= 0 and bear_fvg1.iloc[i - j]:
                bull_since[i] = j
                break

    bull_since = pd.Series(bull_since, index=df.index)
    bull_cond_1 = bull_fvg1 & (bull_since <= lookback_bars)

    bull_since_arr = bull_since.values
    bear_fvg1_arr = bear_fvg1.values
    high_arr = df['high'].values
    low_arr = df['low'].values
    high_2_arr = high_2.values
    low_2_arr = low_2.values

    combined_high_bull = np.full(len(df), np.nan)
    combined_low_bull = np.full(len(df), np.nan)

    for i in range(len(df)):
        if bull_cond_1.iloc[i]:
            bs = int(bull_since_arr[i])
            if i - bs >= 0 and i - bs < len(high_arr):
                combined_high_bull[i] = max(high_arr[i - bs], high_2_arr[i])
            if i - bs + 2 < len(low_arr):
                combined_low_bull[i] = min(low_arr[i - bs + 2], low_arr[i])
            elif i < len(low_arr):
                combined_low_bull[i] = low_arr[i]

    bull_result = bull_cond_1 & (pd.Series(combined_high_bull, index=df.index) - pd.Series(combined_low_bull, index=df.index) >= threshold)

    bear_since = np.full(len(df), np.nan)
    for i in range(len(df)):
        for j in range(1, lookback_bars + 1):
            if i - j >= 0 and bull_fvg1.iloc[i - j]:
                bear_since[i] = j
                break

    bear_since = pd.Series(bear_since, index=df.index)
    bear_cond_1 = bear_fvg1 & (bear_since <= lookback_bars)

    bear_since_arr = bear_since.values

    combined_high_bear = np.full(len(df), np.nan)
    combined_low_bear = np.full(len(df), np.nan)

    for i in range(len(df)):
        if bear_cond_1.iloc[i]:
            bs = int(bear_since_arr[i])
            if i - bs + 2 < len(high_arr):
                combined_low_bear[i] = max(high_arr[i - bs + 2], high_arr[i])
            elif i < len(high_arr):
                combined_low_bear[i] = high_arr[i]
            if i - bs >= 0 and i - bs < len(low_arr):
                combined_high_bear[i] = min(low_arr[i - bs], low_2_arr[i])
            elif i < len(low_arr):
                combined_high_bear[i] = low_arr[i]

    bear_result = bear_cond_1 & (pd.Series(combined_high_bear, index=df.index) - pd.Series(combined_low_bear, index=df.index) >= threshold)

    isWithinMorningWindow = (df['total_minutes'] >= 405) & (df['total_minutes'] < 585)
    isWithinAfternoonWindow = (df['total_minutes'] >= 885) & (df['total_minutes'] < 960)
    in_trading_window = isWithinMorningWindow | isWithinAfternoonWindow

    entries = []
    trade_num = 1

    for i in range(len(df) - 1):
        longSetup = sellsideSweep.iloc[i] or (sellsideSweep.shift(1).fillna(False).iloc[:i+1].any() and (i - sellsideSweep.shift(1).fillna(False).cumsum().idxmax() if sellsideSweep.shift(1).iloc[:i+1].any() else 999) <= fvgWaitBars if i > 0 else False)
        longSetup = sellsideSweep.iloc[i]
        longEntry = longSetup and bullishFVGValid.iloc[i] and in_trading_window.iloc[i] and bull_result.iloc[i]

        shortSetup = buysideSweep.iloc[i]
        shortEntry = shortSetup and bearishFVGValid.iloc[i] and in_trading_window.iloc[i] and bear_result.iloc[i]

        if longEntry:
            entry_ts = int(df['ts'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

        if shortEntry:
            entry_ts = int(df['ts'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries