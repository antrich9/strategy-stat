import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    open_prices = df['open']
    high_prices = df['high']
    low_prices = df['low']
    close_prices = df['close']
    volume = df['volume']
    timestamps = df['time']
    
    entries = []
    trade_num = 1
    
    is_up = close_prices > open_prices
    is_down = close_prices < open_prices
    
    ob_up = is_down.shift(1) & is_up & (close_prices > high_prices.shift(1))
    ob_down = is_up.shift(1) & is_down & (close_prices < low_prices.shift(1))
    
    fvg_up = low_prices > high_prices.shift(2)
    fvg_down = high_prices < low_prices.shift(2)
    
    vol_filt = volume.shift(1) > volume.rolling(9).mean() * 1.5
    
    tr1 = high_prices - low_prices.shift(2)
    tr2 = high_prices.shift(2) - low_prices
    tr = pd.concat([tr1, tr2], axis=1).max(axis=1)
    atr = tr.rolling(20).mean()
    atr_filt = (low_prices - high_prices.shift(2) > atr / 1.5) | (low_prices.shift(2) - high_prices > atr / 1.5)
    
    loc = close_prices.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    loc_filt_b = loc2
    loc_filt_s = ~loc2
    
    bfvg = fvg_up & vol_filt & atr_filt & loc_filt_b
    sfvg = fvg_down & vol_filt & atr_filt & loc_filt_s
    
    stacked_ob_fvg_bull = ob_up.shift(1) & fvg_up
    stacked_ob_fvg_bear = ob_down.shift(1) & fvg_down
    
    utc_times = pd.to_datetime(timestamps, unit='ms', utc=True)
    hours = utc_times.dt.hour
    minutes = utc_times.dt.minute
    
    in_window_1 = (hours >= 7) & (hours <= 10) & ~((hours == 10) & (minutes > 59))
    in_window_2 = (hours >= 15) & (hours <= 16) & ~((hours == 16) & (minutes > 59))
    in_trading_window = in_window_1 | in_window_2
    
    long_condition = (bfvg | stacked_ob_fvg_bull) & in_trading_window
    short_condition = (sfvg | stacked_ob_fvg_bear) & in_trading_window
    
    for i in range(len(df)):
        if np.isnan(atr.iloc[i]) or np.isnan(loc.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            ts = int(timestamps.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close_prices.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_prices.iloc[i]),
                'raw_price_b': float(close_prices.iloc[i])
            })
            trade_num += 1
        
        if short_condition.iloc[i]:
            ts = int(timestamps.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close_prices.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close_prices.iloc[i]),
                'raw_price_b': float(close_prices.iloc[i])
            })
            trade_num += 1
    
    return entries