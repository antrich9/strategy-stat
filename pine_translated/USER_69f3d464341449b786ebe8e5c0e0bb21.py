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
    trade_num = 1
    
    close = df['close']
    high = df['high']
    low = df['low']
    open_col = df['open']
    time_col = df['time']
    
    # Wilder ATR
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # Parameters
    tenkan_len, tenkan_mult = 9, 2.0
    kijun_len, kijun_mult = 26, 4.0
    spanB_len, spanB_mult = 52, 6.0
    
    hl2 = (high + low) / 2
    
    def calc_avg(src, length, mult):
        at = atr * mult
        up = hl2 + at
        dn = hl2 - at
        
        upper = pd.Series(index=src.index, dtype=float)
        lower = pd.Series(index=src.index, dtype=float)
        os = pd.Series(0, index=src.index, dtype=int)
        
        upper.iloc[0] = up.iloc[0]
        lower.iloc[0] = dn.iloc[0]
        
        for i in range(1, len(src)):
            upper.iloc[i] = up.iloc[i] if src.iloc[i-1] >= upper.iloc[i-1] else min(up.iloc[i], upper.iloc[i-1])
            lower.iloc[i] = dn.iloc[i] if src.iloc[i-1] <= lower.iloc[i-1] else max(dn.iloc[i], lower.iloc[i-1])
            os.iloc[i] = 1 if src.iloc[i] > upper.iloc[i] else (0 if src.iloc[i] < lower.iloc[i] else os.iloc[i-1])
        
        spt = pd.Series(index=src.index, dtype=float)
        max_arr = pd.Series(index=src.index, dtype=float)
        min_arr = pd.Series(index=src.index, dtype=float)
        
        spt = pd.where(os == 1, lower, upper)
        
        max_arr.iloc[0] = src.iloc[0]
        min_arr.iloc[0] = src.iloc[0]
        
        for i in range(1, len(src)):
            cross = (src.iloc[i] - spt.iloc[i]) * (src.iloc[i-1] - spt.iloc[i-1]) < 0
            if cross:
                max_arr.iloc[i] = max(src.iloc[i], max_arr.iloc[i-1])
                min_arr.iloc[i] = min(src.iloc[i], min_arr.iloc[i-1])
            elif os.iloc[i] == 1:
                max_arr.iloc[i] = max(src.iloc[i], max_arr.iloc[i-1])
                min_arr.iloc[i] = min(min_arr.iloc[i-1], spt.iloc[i])
            else:
                max_arr.iloc[i] = max(max_arr.iloc[i-1], spt.iloc[i])
                min_arr.iloc[i] = min(src.iloc[i], min_arr[i-1])
        
        return (max_arr + min_arr) / 2
    
    tenkan = calc_avg(close, tenkan_len, tenkan_mult)
    kijun = calc_avg(close, kijun_len, kijun_mult)
    senkouA = (tenkan + kijun) / 2
    senkouB = calc_avg(close, spanB_len, spanB_mult)
    
    ema100 = close.ewm(span=100, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    
    cum_vol = df['volume'].cumsum()
    if cum_vol.iloc[-1] == 0:
        return results
    
    vwap_numerator = (df['volume'] * (high + low + close) / 3).cumsum()
    vwap_denominator = df['volume'].cumsum()
    vwap = vwap_numerator / vwap_denominator
    
    # Time filter - Convert to local time checks
    london_start_hour = 7
    london_end_hour = 10
    ny_start_hour = 14
    ny_end_hour = 17
    
    ts_series = pd.to_datetime(time_col, unit='s', utc=True)
    hours = ts_series.dt.hour
    
    in_london = (hours >= london_start_hour) & (hours < london_end_hour)
    in_ny = (hours >= ny_start_hour) & (hours < ny_end_hour)
    in_trading_window = in_london | in_ny
    
    # Crossover/crossunder
    tenkan_above_kijun = tenkan > kijun
    tenkan_below_kijun = tenkan < kijun
    
    crossover_tenkan_kijun = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
    crossunder_tenkan_kijun = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))
    
    # Entry conditions
    longCondition = (close > ema200) & (close > ema100) & (close > vwap) & crossover_tenkan_kijun
    shortCondition = (close < ema200) & (close < ema100) & (close < vwap) & crossunder_tenkan_kijun
    
    # Skip bars with NaN in required indicators
    valid_idx = ~(tenkan.isna() | kijun.isna() | ema100.isna() | ema200.isna() | vwap.isna())
    
    prev_position_open = False
    
    for i in range(len(df)):
        if not valid_idx.iloc[i]:
            continue
        if not in_trading_window.iloc[i]:
            continue
        
        if prev_position_open:
            continue
        
        entry_price = close.iloc[i]
        ts = int(time_col.iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        if longCondition.iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            prev_position_open = True
        
        elif shortCondition.iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            prev_position_open = True
    
    return results