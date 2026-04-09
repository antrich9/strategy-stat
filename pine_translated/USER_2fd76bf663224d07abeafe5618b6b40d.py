import pandas as pd
import numpy as np
from datetime import datetime, timezone

def wilder_atr(high, low, close, length):
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/length, adjust=False).mean()
    return atr

def generate_entries(df: pd.DataFrame) -> list:
    entries = []
    trade_num = 1
    
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)
    
    bullG = df['low'] > prev_high
    bearG = df['high'] < prev_low
    
    atr = wilder_atr(df['high'], df['low'], df['close'], 144)
    atr_val = atr * 0.5
    
    bull_fvg_upper = np.where((df['low'] - df['high'].shift(2)) > atr_val, df['high'].shift(2), np.nan)
    bull_fvg_lower = np.where((df['low'] - df['high'].shift(2)) > atr_val, df['low'], np.nan)
    bear_fvg_upper = np.where((df['low'].shift(2) - df['high']) > atr_val, df['high'], np.nan)
    bear_fvg_lower = np.where((df['low'].shift(2) - df['high']) > atr_val, df['low'].shift(2), np.nan)
    
    bull_midpoint = pd.Series(np.where(~np.isnan(bull_fvg_upper), (bull_fvg_upper + bull_fvg_lower) / 2, np.nan), index=df.index)
    bear_midpoint = pd.Series(np.where(~np.isnan(bear_fvg_upper), (bear_fvg_upper + bear_fvg_lower) / 2, np.nan), index=df.index)
    
    df_ts = pd.to_datetime(df['time'], unit='s', utc=True)
    morning_start = df_ts.dt.strftime('%Y-%m-%d').apply(lambda x: datetime(int(x[:4]), int(x[5:7]), int(x[8:10]), 7, 45, tzinfo=timezone.utc))
    morning_end = df_ts.dt.strftime('%Y-%m-%d').apply(lambda x: datetime(int(x[:4]), int(x[5:7]), int(x[8:10]), 9, 45, tzinfo=timezone.utc))
    afternoon_start = df_ts.dt.strftime('%Y-%m-%d').apply(lambda x: datetime(int(x[:4]), int(x[5:7]), int(x[8:10]), 14, 45, tzinfo=timezone.utc))
    afternoon_end = df_ts.dt.strftime('%Y-%m-%d').apply(lambda x: datetime(int(x[:4]), int(x[5:7]), int(x[8:10]), 16, 45, tzinfo=timezone.utc))
    is_within_morning = (df_ts >= morning_start) & (df_ts < morning_end)
    is_within_afternoon = (df_ts >= afternoon_start) & (df_ts < afternoon_end)
    in_trading_window = is_within_morning | is_within_afternoon
    
    dow = df_ts.dt.dayofweek
    is_thursday = dow == 3
    is_friday = dow == 4
    is_thursday_morning = is_thursday & is_within_morning
    is_friday_morning = is_friday & is_within_morning
    
    daily_df = df.copy()
    daily_df['date'] = df_ts.dt.date
    daily_high = daily_df.groupby('date')['high'].max().reindex(daily_df['date']).values
    daily_low = daily_df.groupby('date')['low'].min().reindex(daily_df['date']).values
    prev_day_high = np.concatenate([[np.nan], daily_high[:-1]])
    prev_day_low = np.concatenate([[np.nan], daily_low[:-1]])
    flagpdh = df['high'] > prev_day_high
    flagpdl = df['low'] < prev_day_low
    
    bull_crossunder_midpoint = (df['low'].shift(1) > bull_midpoint.shift(1)) & (df['low'] < bull_midpoint) & ~bull_midpoint.isna() & ~bull_midpoint.shift(1).isna()
    bear_crossunder_midpoint = (df['high'].shift(1) < bear_midpoint.shift(1)) & (df['high'] > bear_midpoint) & ~bear_midpoint.isna() & ~bear_midpoint.shift(1).isna()
    
    bull_fvg_active = pd.Series(False, index=df.index)
    bear_fvg_active = pd.Series(False, index=df.index)
    last_was_bull = False
    
    for i in range(len(df)):
        if not pd.isna(bull_fvg_upper[i]):
            bull_fvg_active.iloc[i] = True
            last_was_bull = True
        elif not pd.isna(bear_fvg_upper[i]):
            bear_fvg_active.iloc[i] = True
            last_was_bull = False
        
        has_fvg = bull_fvg_active.iloc[i] or bear_fvg_active.iloc[i]
        
        if i > 0 and bull_fvg_active.iloc[i-1]:
            pct = (df['low'].iloc[i] - df['high'].shift(2).iloc[i]) / (df['high'].shift(2).iloc[i] - bull_fvg_lower[i]) if not pd.isna(bull_fvg_lower[i]) and df['high'].shift(2).iloc[i] != bull_fvg_lower[i] else 1.0
            bull_fvg_active.iloc[i] = pct > 0.01
        elif i > 0 and bear_fvg_active.iloc[i-1]:
            pct = (bear_fvg_upper[i] - df['high'].iloc[i]) / (bear_fvg_upper[i] - df['low'].shift(2).iloc[i]) if not pd.isna(bear_fvg_upper[i]) and bear_fvg_upper[i] != df['low'].shift(2).iloc[i] else 1.0
            bear_fvg_active.iloc[i] = pct > 0.01
        
        long_condition = (bull_fvg_active.iloc[i] and last_was_bull and bull_crossunder_midpoint.iloc[i] and in_trading_window.iloc[i] and flagpdl.iloc[i] and not is_thursday_morning.iloc[i] and not is_friday_morning.iloc[i])
        short_condition = (bear_fvg_active.iloc[i] and not last_was_bull and bear_crossunder_midpoint.iloc[i] and in_trading_window.iloc[i] and flagpdh.iloc[i] and not is_thursday_morning.iloc[i] and not is_friday_morning.iloc[i])
        
        if long_condition:
            ts = int(df['time'].iloc[i])
            entries.append({'trade_num': trade_num, 'direction': 'long', 'entry_ts': ts, 'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(), 'entry_price_guess': bull_midpoint.iloc[i] if not pd.isna(bull_midpoint.iloc[i]) else float(df['close'].iloc[i]), 'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0, 'raw_price_a': float(df['close'].iloc[i]), 'raw_price_b': float(df['close'].iloc[i])})
            trade_num += 1
        elif short_condition:
            ts = int(df['time'].iloc[i])
            entries.append({'trade_num': trade_num, 'direction': 'short', 'entry_ts': ts, 'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(), 'entry_price_guess': bear_midpoint.iloc[i] if not pd.isna(bear_midpoint.iloc[i]) else float(df['close'].iloc[i]), 'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0, 'raw_price_a': float(df['close'].iloc[i]), 'raw_price_b': float(df['close'].iloc[i])})
            trade_num += 1
    
    return entries