import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Time window check function
    def is_within_time_window(ts):
        try:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            h, m = dt.hour, dt.minute
            morning_start = (7, 45)
            morning_end = (9, 45)
            afternoon_start = (14, 45)
            afternoon_end = (16, 45)
            
            in_morning = (h > morning_start[0] or (h == morning_start[0] and m >= morning_start[1])) and \
                        (h < morning_end[0] or (h == morning_end[0] and m < morning_end[1]))
            in_afternoon = (h > afternoon_start[0] or (h == afternoon_start[0] and m >= afternoon_start[1])) and \
                          (h < afternoon_end[0] or (h == afternoon_end[0] and m < afternoon_end[1]))
            return in_morning or in_afternoon
        except:
            return False
    
    isWithinTimeWindow = df['time'].apply(is_within_time_window)
    
    # Input parameters
    inp1 = False
    inp2 = False
    inp3 = False
    
    # Volume filter
    volfilt = (df['volume'].shift(1) > df['volume'].rolling(9).mean() * 1.5) if inp1 else pd.Series(True, index=df.index)
    
    # ATR filter (Wilder method: ewm with alpha=1/period)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, min_periods=20, adjust=False).mean()
    atrfilt = ((df['low'] - df['high'].shift(2) > atr/1.5) | (df['low'].shift(2) - df['high'] > atr/1.5)) if inp2 else pd.Series(True, index=df.index)
    
    # Trend filter
    loc = df['close'].ewm(span=54, adjust=False).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2 if inp3 else pd.Series(True, index=df.index)
    locfilts = ~loc2 if inp3 else pd.Series(True, index=df.index)
    
    # OB conditions
    is_up = df['close'] > df['open']
    is_down = df['close'] < df['open']
    
    obUp = is_down.shift(1) & is_up & (df['close'] > df['high'].shift(1))
    obDown = is_up.shift(1) & is_down & (df['close'] < df['low'].shift(1))
    
    # FVG conditions
    fvgUp = df['low'] > df['high'].shift(2)
    fvgDown = df['high'] < df['low'].shift(2)
    
    # Stacked condition
    stacked_bull = obUp | fvgUp
    stacked_bear = obDown | fvgDown
    
    # FVG filter conditions
    bfvg = df['low'] > df['high'].shift(2) & volfilt & atrfilt & locfiltb
    sfvg = df['high'] < df['low'].shift(2) & volfilt & atrfilt & locfilts
    
    # Fill NaN values
    bfvg = bfvg.fillna(False)
    sfvg = sfvg.fillna(False)
    stacked_bull = stacked_bull.fillna(False)
    stacked_bear = stacked_bear.fillna(False)
    volfilt = volfilt.fillna(True)
    atrfilt = atrfilt.fillna(True)
    locfiltb = locfiltb.fillna(True)
    locfilts = locfilts.fillna(True)
    
    entries = []
    trade_num = 0
    
    for i in range(5, len(df)):
        if pd.isna(df['time'].iloc[i]):
            continue
        
        bull_entry = stacked_bull.iloc[i] and bfvg.iloc[i]
        bear_entry = stacked_bear.iloc[i] and sfvg.iloc[i]
        
        if bull_entry or bear_entry:
            if isWithinTimeWindow.iloc[i]:
                direction = 'long' if bull_entry else 'short'
                trade_num += 1
                ts = df['time'].iloc[i]
                entries.append({
                    'trade_num': trade_num,
                    'direction': direction,
                    'entry_ts': int(ts),
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': df['close'].iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': df['close'].iloc[i],
                    'raw_price_b': df['close'].iloc[i]
                })
    
    return entries