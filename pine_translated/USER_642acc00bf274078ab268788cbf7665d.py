import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    result = []
    trade_num = 0
    
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    vol_sma = volume.rolling(9).mean()
    vol_filt = volume.shift(1) > vol_sma * 1.5
    
    loc = close.rolling(54).mean()
    bull_loc_filt = loc > loc.shift(1)
    bear_loc_filt = loc < loc.shift(1)
    
    bull_fvg = (low > high.shift(2)) & vol_filt & (atr > 0) & bull_loc_filt
    bear_fvg = (high < low.shift(2)) & vol_filt & (atr > 0) & bear_loc_filt
    
    bull_top_imbalance_bway = (low.shift(2) <= open.shift(1)) & (high >= close.shift(1)) & (close < low.shift(1))
    bear_top_imbalance_xbway = (low.shift(2) <= open.shift(1)) & (high >= close.shift(1)) & (close > low.shift(1))
    imbalance_size = low.shift(2) - high
    
    dt_series = pd.to_datetime(df['time'], unit='s', utc=True)
    hour = dt_series.dt.hour
    minute = dt_series.dt.minute
    
    morning_window = ((hour == 7) & (minute >= 45)) | ((hour.isin([8, 9])) & ~((hour == 9) & (minute > 45)))
    afternoon_window = ((hour == 14) & (minute >= 45)) | ((hour.isin([15, 16])) & ~((hour == 16) & (minute > 45)))
    in_time_window = morning_window | afternoon_window
    
    for i in range(2, len(df)):
        if bull_fvg.iloc[i] and bull_top_imbalance_bway.iloc[i] and imbalance_size.iloc[i] > 0 and in_time_window.iloc[i]:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            result.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
        
        if bear_fvg.iloc[i] and bear_top_imbalance_xbway.iloc[i] and imbalance_size.iloc[i] > 0 and in_time_window.iloc[i]:
            trade_num += 1
            ts = int(df['time'].iloc[i])
            result.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
    
    return result