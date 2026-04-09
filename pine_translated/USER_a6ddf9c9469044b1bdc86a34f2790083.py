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
    
    # Strategy parameters (from input defaults)
    usePrevDay = True
    sweepBuffer = 2
    lookbackSw = 5
    waitForCHoCH = True
    chochWindow = 10
    minFvgTicks = 3
    fvgWindow = 10
    atrLength = 14
    
    # Constants
    NA_VALUE = 999
    
    # Ensure proper types
    df = df.copy()
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    df['time'] = df['time'].astype(int)
    
    n = len(df)
    if n < 3:
        return []
    
    # Calculate min_tick (smallest price movement)
    price_diff = df['high'] - df['low']
    price_diff = price_diff[price_diff > 0]
    min_tick = price_diff.min() if len(price_diff) > 0 else 0.0001
    buffer_ticks = sweepBuffer * min_tick
    
    # Calculate Previous Day High/Low using resampling to daily
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df_daily = df.set_index('datetime').resample('D').agg({'high': 'max', 'low': 'min'}).dropna()
    
    prev_high = pd.Series(index=df.index, dtype=float)
    prev_low = pd.Series(index=df.index, dtype=float)
    
    for idx in df.index:
        current_ts = df.loc[idx, 'datetime']
        current_date = current_ts.date()
        
        # Get previous day's high/low
        daily_dates = df_daily.index.to_series().dt.date
        prev_day_mask = daily_dates < current_date
        
        if prev_day_mask.any():
            prev_day_dates = daily_dates[prev_day_mask]
            if len(prev_day_dates) > 0:
                prev_date = prev_day_dates.max()
                prev_day_data = df_daily[df_daily.index.to_series().dt.date == prev_date]
                if not prev_day_data.empty:
                    prev_high.loc[idx] = prev_day_data['high'].iloc[-1]
                    prev_low.loc[idx] = prev_day_data['low'].iloc[-1]
    
    # Sweep Detection
    sweep_high = usePrevDay & ~prev_high.isna() & (df['high'] > prev_high + buffer_ticks) & (df['close'] < prev_high)
    sweep_low = usePrevDay & ~prev_low.isna() & (df['low'] < prev_low - buffer_ticks) & (df['close'] > prev_low)
    
    # CHoCH Detection (Pivot Highs/Lows)
    pivots_high = pd.Series(index=df.index, dtype=float)
    pivots_low = pd.Series(index=df.index, dtype=float)
    lb = lookbackSw
    
    for i in range(lb, n - lb):
        is_local_max = True
        is_local_min = True
        
        for j in range(1, lb + 1):
            if df['high'].iloc[i] <= df['high'].iloc[i - j]:
                is_local_max = False
            if df['high'].iloc[i] <= df['high'].iloc[i + j]:
                is_local_max = False
            if df['low'].iloc[i] >= df['low'].iloc[i - j]:
                is_local_min = False
            if df['low'].iloc[i] >= df['low'].iloc[i + j]:
                is_local_min = False
        
        if is_local_max:
            pivots_high.iloc[i] = df['high'].iloc[i]
        if is_local_min:
            pivots_low.iloc[i] = df['low'].iloc[i]
    
    # Forward fill to track last swing high/low
    last_swing_high = pivots_high.where(pivots_high.notna()).ffill()
    last_swing_low = pivots_low.where(pivots_low.notna()).ffill()
    
    # CHoCH signals
    choch_up = last_swing_high.notna() & (df['close'] > last_swing_high) & (df['close'].shift(1) <= last_swing_high)
    choch_down = last_swing_low.notna() & (df['close'] < last_swing_low) & (df['close'].shift(1) >= last_swing_low)
    
    # FVG Detection
    min_tick_val = max(min_tick, 1e-10)
    fvg_bull = (df['low'].shift(1) > df['high']) & ((df['low'].shift(1) - df['high']) / min_tick_val >= minFvgTicks)
    fvg_bear = (df['high'].shift(1) < df['low']) & ((df['low'] - df['high'].shift(1)) / min_tick_val >= minFvgTicks)
    
    # State tracking variables
    last_sweep_high_bar = pd.Series(index=df.index, dtype=float)
    last_sweep_low_bar = pd.Series(index=df.index, dtype=float)
    last_choch_up_bar = pd.Series(index=df.index, dtype=float)
    last_choch_down_bar = pd.Series(index=df.index, dtype=float)
    
    # Initialize all bars as NaN
    for ser in [last_sweep_high_bar, last_sweep_low_bar, last_choch_up_bar, last_choch_down_bar]:
        ser.iloc[:] = np.nan
    
    # Track events through time
    for i in range(2, n):
        if i > 0:
            last_sweep_high_bar.iloc[i] = last_sweep_high_bar.iloc[i - 1]
            last_sweep_low_bar.iloc[i] = last_sweep_low_bar.iloc[i - 1]
            last_choch_up_bar.iloc[i] = last_choch_up_bar.iloc[i - 1]
            last_choch_down_bar.iloc[i] = last_choch_down_bar.iloc[i - 1]
        
        if sweep_high.iloc[i] if hasattr(sweep_high, 'iloc') else sweep_high[i]:
            last_sweep_high_bar.iloc[i] = float(i)
        if sweep_low.iloc[i] if hasattr(sweep_low, 'iloc') else sweep_low[i]:
            last_sweep_low_bar.iloc[i] = float(i)
        if choch_up.iloc[i] if hasattr(choch_up, 'iloc') else choch_up[i]:
            last_choch_up_bar.iloc[i] = float(i)
        if choch_down.iloc[i] if hasattr(choch_down, 'iloc') else choch_down[i]:
            last_choch_down_bar.iloc[i] = float(i)
        
        bars_since_sweep_low = i - last_sweep_low_bar.iloc[i] if not np.isnan(last_sweep_low_bar.iloc[i]) else NA_VALUE
        bars_since_sweep_high = i - last_sweep_high_bar.iloc[i] if not np.isnan(last_sweep_high_bar.iloc[i]) else NA_VALUE
        bars_since_choch_up = i - last_choch_up_bar.iloc[i] if not np.isnan(last_choch_up_bar.iloc[i]) else NA_VALUE
        bars_since_choch_down = i - last_choch_down_bar.iloc[i] if not np.isnan(last_choch_down_bar.iloc[i]) else NA_VALUE
        
        if bars_since_sweep_low > chochWindow + fvgWindow:
            last_sweep_low_bar.iloc[i] = np.nan
        if bars_since_sweep_high > chochWindow + fvgWindow:
            last_sweep_high_bar.iloc[i] = np.nan
        if bars_since_choch_up > fvgWindow:
            last_choch_up_bar.iloc[i] = np.nan
        if bars_since_choch_down > fvgWindow:
            last_choch_down_bar.iloc[i] = np.nan
    
    # Build entry conditions
    entries_long = pd.Series(False, index=df.index)
    entries_short = pd.Series(False, index=df.index)
    
    for i in range(2, n):
        bars_ssl = i - last_sweep_low_bar.iloc[i] if not np.isnan(last_sweep_low_bar.iloc[i]) else NA_VALUE
        bars_ssh = i - last_sweep_high_bar.iloc[i] if not np.isnan(last_sweep_high_bar.iloc[i]) else NA_VALUE
        bars_scu = i - last_choch_up_bar.iloc[i] if not np.isnan(last_choch_up_bar.iloc[i]) else NA_VALUE
        bars_scd = i - last_choch_down_bar.iloc[i] if not np.isnan(last_choch_down_bar.iloc[i]) else NA_VALUE
        
        is_sl = bool(sweep_low.iloc[i]) if hasattr(sweep_low, 'iloc') else False
        is_sh = bool(sweep_high.iloc[i]) if hasattr(sweep_high, 'iloc') else False
        is_fvg_bull = bool(fvg_bull.iloc[i]) if hasattr(fvg_bull, 'iloc') else False
        is_fvg_bear = bool(fvg_bear.iloc[i]) if hasattr(fvg_bear, 'iloc') else False
        is_choch_up = bool(choch_up.iloc[i]) if hasattr(choch_up, 'iloc') else False
        is_choch_down = bool(choch_down.iloc[i]) if hasattr(choch_down, 'iloc') else False
        
        lsw_ok = bars_ssl > 0 and bars_ssl <= chochWindow + fvgWindow
        lch_ok = bars_scu > 0 and bars_scu <= fvgWindow
        lch_after_sl = not np.isnan(last_sweep_low_bar.iloc[i]) and not np.isnan(last_choch_up_bar.iloc[i]) and last_choch_up_bar.iloc[i] > last_sweep_low_bar.iloc[i]
        
        if waitForCHoCH:
            long_setup = lsw_ok and lch_ok and lch_after_sl and is_fvg_bull
        else:
            long_setup = (is_sl or lsw_ok) and is_fvg_bull
        
        ssw_ok = bars_ssh > 0 and bars_ssh <= chochWindow + fvgWindow
        sch_ok = bars_scd > 0 and bars_scd <= fvgWindow
        sch_after_sh = not np.isnan(last_sweep_high_bar.iloc[i]) and not np.isnan(last_choch_down_bar.iloc[i]) and last_choch_down_bar.iloc[i] > last_sweep_high_bar.iloc[i]
        
        if waitForCHoCH:
            short_setup = ssw_ok and sch_ok and sch_after_sh and is_fvg_bear
        else:
            short_setup = (is_sh or ssw_ok) and is_fvg_bear
        
        if long_setup:
            entries_long.iloc[i] = True
        if short_setup:
            entries_short.iloc[i] = True
    
    # Generate output
    results = []
    trade_num = 1
    
    for i in df.index:
        ts = int(df['time'].iloc[i])
        if entries_long.iloc[i] if hasattr(entries_long, 'iloc') else entries_long[i]:
            entry_price = float(df['close'].iloc[i])
            results.append({
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
    
    for i in df.index:
        ts = int(df['time'].iloc[i])
        if entries_short.iloc[i] if hasattr(entries_short, 'iloc') else entries_short[i]:
            entry_price = float(df['close'].iloc[i])
            results.append({
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
    
    return results