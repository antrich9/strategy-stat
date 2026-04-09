import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    high = df['high']
    low = df['low']
    close = df['close']
    time = df['time']
    
    bullTop = high
    bullBottom = low.shift(2)
    bullFVG = bullBottom.shift(2) > high
    
    bearTop = high.shift(2)
    bearBottom = low
    bearFVG = bearTop.shift(2) < low
    
    swingHigh = high.rolling(20).max()
    swingLow = low.rolling(20).min()
    
    in_current_range = (bullTop <= swingHigh) & (bullBottom >= swingLow) | \
                      (bearTop <= swingHigh) & (bearBottom >= swingLow)
    
    dt = pd.to_datetime(time, unit='s', utc=True).dt.tz_convert('Europe/London')
    hour = dt.dt.hour
    minute = dt.dt.minute
    is_between_3pm_and_4pm = (hour == 15) | ((hour == 16) & (minute == 0))
    
    bullish_top = pd.Series(np.nan, index=df.index)
    bullish_bottom = pd.Series(np.nan, index=df.index)
    bearish_top = pd.Series(np.nan, index=df.index)
    bearish_bottom = pd.Series(np.nan, index=df.index)
    fvg_created = pd.Series(False, index=df.index)
    
    for i in range(2, len(df)):
        if is_between_3pm_and_4pm.iloc[i]:
            if i == 0 or not is_between_3pm_and_4pm.iloc[i-1]:
                bullish_top = pd.Series(np.nan, index=df.index)
                bullish_bottom = pd.Series(np.nan, index=df.index)
                bearish_top = pd.Series(np.nan, index=df.index)
                bearish_bottom = pd.Series(np.nan, index=df.index)
                fvg_created = pd.Series(False, index=df.index)
            
            if bullFVG.iloc[i] and in_current_range.iloc[i] and bullish_top.isna().iloc[i]:
                bullish_top.iloc[i] = bullTop.iloc[i]
                bullish_bottom.iloc[i] = bullBottom.iloc[i]
                fvg_created.iloc[i] = True
            
            if not fvg_created.iloc[i] and bearFVG.iloc[i] and in_current_range.iloc[i] and bearish_bottom.isna().iloc[i]:
                bearish_top.iloc[i] = bearTop.iloc[i]
                bearish_bottom.iloc[i] = bearBottom.iloc[i]
                fvg_created.iloc[i] = True
    
    bullish_entry = is_between_3pm_and_4pm & bullish_top.notna() & \
                    (low.shift(1) > bullish_top) & (low <= bullish_top)
    
    bearish_entry = is_between_3pm_and_4pm & bearish_top.notna() & \
                    (high.shift(1) < bearish_bottom) & (high >= bearish_bottom)
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if bullish_entry.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(time.iloc[i]),
                'entry_time': datetime.fromtimestamp(time.iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': bullish_top.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': bullish_top.iloc[i],
                'raw_price_b': bullish_top.iloc[i]
            })
            trade_num += 1
        elif bearish_entry.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(time.iloc[i]),
                'entry_time': datetime.fromtimestamp(time.iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': bearish_bottom.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': bearish_bottom.iloc[i],
                'raw_price_b': bearish_bottom.iloc[i]
            })
            trade_num += 1
    
    return entries