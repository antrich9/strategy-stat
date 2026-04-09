import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    open_price = df['open']
    high_price = df['high']
    low_price = df['low']
    close_price = df['close']
    volume_data = df['volume']
    timestamps = df['time']
    
    sma_9 = volume_data.rolling(9).mean()
    atr_raw = ta_atr(high_price, low_price, close_price)
    atr = atr_raw / 1.5
    sma_54 = close_price.rolling(54).mean()
    loc = sma_54
    loc_shift = loc.shift(1)
    loc2 = loc > loc_shift
    volfilt = sma_9 * 1.5
    atrfilt = ((low_price - high_price.shift(2) > atr) | (low_price.shift(2) - high_price > atr))
    locfiltb = loc2
    locfilts = ~loc2
    bfvg = (low_price > high_price.shift(2)) & (volume_data > volfilt) & atrfilt & locfiltb
    sfvg = (high_price < low_price.shift(2)) & (volume_data > volfilt) & atrfilt & locfilts
    obUp = isObUp(close_price, open_price, high_price, low_price)
    obDown = isObDown(close_price, open_price, high_price, low_price)
    fvgUp = low_price > high_price.shift(2)
    fvgDown = high_price < low_price.shift(2)
    bullStack = obUp & fvgUp
    bearStack = obDown & fvgDown
    
    dt = pd.to_datetime(timestamps, unit='ms')
    hour = dt.dt.hour
    minute = dt.dt.minute
    inMorningWindow = (((hour == 7) & (minute >= 45)) | ((hour == 8) & (minute < 45)))
    inAfternoonWindow = (((hour == 14) & (minute >= 45)) | ((hour == 15) & (minute < 45)))
    in_time_window = inMorningWindow | inAfternoonWindow
    
    conditions = [
        ('long', bfvg),
        ('short', sfvg),
        ('long', bullStack),
        ('short', bearStack)
    ]
    
    trade_num = 1
    trades = []
    for direction, cond in conditions:
        for i in range(len(df)):
            if cond.iloc[i] and in_time_window.iloc[i]:
                entry_price = close_price.iloc[i]
                ts = int(timestamps.iloc[i])
                trades.append({
                    'trade_num': trade_num,
                    'direction': direction,
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
    return trades

def ta_atr(high, low, close):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean