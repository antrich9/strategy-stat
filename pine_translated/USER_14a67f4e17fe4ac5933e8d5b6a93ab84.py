import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    entries = []
    trade_num = 0
    
    ts = df['time'].values
    dt_utc = pd.to_datetime(ts, unit='s', utc=True)
    dt_london = dt_utc.dt.tz_convert('Europe/London')
    hour = dt_london.hour
    minute = dt_london.minute
    
    morning_window = (hour == 8) | ((hour == 9) & (minute <= 55))
    afternoon_window = (hour >= 14) & (hour < 17) | ((hour == 16) & (minute <= 55))
    is_within_window = morning_window | afternoon_window
    
    bull_fvg = (df['low'] >= df['high'].shift(2))
    bear_fvg = (df['low'].shift(2) >= df['high'])
    
    bull_fvg_top = pd.Series(np.where(bull_fvg, df['high'].shift(2), np.nan), index=df.index)
    bear_fvg_bottom = pd.Series(np.where(bear_fvg, df['low'].shift(2), np.nan), index=df.index)
    
    for i in range(2, len(df)):
        if is_within_window[i]:
            if not np.isnan(bull_fvg_top.iloc[i]):
                bull_top = bull_fvg_top.iloc[i]
                curr_low = df['low'].iloc[i]
                prev_low = df['low'].iloc[i - 1]
                if curr_low < bull_top and prev_low >= bull_top:
                    trade_num += 1
                    entry_ts = int(df['time'].iloc[i])
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': entry_ts,
                        'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                        'entry_price_guess': float(df['close'].iloc[i]),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': bull_top,
                        'raw_price_b': bull_top
                    })
            
            if not np.isnan(bear_fvg_bottom.iloc[i]):
                bear_bottom = bear_fvg_bottom.iloc[i]
                curr_close = df['close'].iloc[i]
                prev_close = df['close'].iloc[i - 1]
                if curr_close > bear_bottom and prev_close <= bear_bottom:
                    trade_num += 1
                    entry_ts = int(df['time'].iloc[i])
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': entry_ts,
                        'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                        'entry_price_guess': float(df['close'].iloc[i]),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': bear_bottom,
                        'raw_price_b': bear_bottom
                    })
    
    return entries