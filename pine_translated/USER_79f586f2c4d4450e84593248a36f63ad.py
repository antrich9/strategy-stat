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
    
    # Prepare columns
    close = df['close']
    high = df['high']
    low = df['low']
    open_col = df['open']
    volume = df['volume']
    time = df['time']
    
    # Helper functions for OB and FVG detection
    def isUp(idx):
        return close.iloc[idx] > open_col.iloc[idx]
    
    def isDown(idx):
        return close.iloc[idx] < open_col.iloc[idx]
    
    # Order Block conditions
    # isObUp(1): previous bar is down, current is up, current close > previous high
    # isObDown(1): previous bar is up, current is down, current close < previous low
    obUp = isDown(1) & isUp(0) & (close.iloc[0] > high.iloc[1])
    obDown = isUp(1) & isDown(0) & (close.iloc[0] < low.iloc[1])
    
    # Fair Value Gap conditions
    # isFvgUp(0): current low > high[2]
    # isFvgDown(0): current high < low[2]
    fvgUp = low > high.shift(2)
    fvgDown = high < low.shift(2)
    
    # Volume filter: volume[1] > sma(volume, 9) * 1.5
    vol_sma = volume.rolling(9).mean()
    volfilt = volume.shift(1) > vol_sma * 1.5
    
    # ATR filter: (low - high[2] > atr) or (low[2] - high > atr), where atr = ta.atr(20) / 1.5
    def calc_wilder_atr(High, Low, Close, length=20):
        tr1 = High - Low
        tr2 = np.abs(High - Close.shift(1))
        tr3 = np.abs(Low - Close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/length, adjust=False).mean()
        return atr
    
    atr_raw = calc_wilder_atr(high, low, close, 20)
    atr_filter = atr_raw / 1.5
    atrfilt = (low - high.shift(2) > atr_filter) | (low.shift(2) - high > atr_filter)
    
    # Trend filter: SMA 54 slope
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # Combined FVG conditions
    bfvg = fvgUp & volfilt & atrfilt & locfiltb
    sfvg = fvgDown & volfilt & atrfilt & locfilts
    
    # Time window filter (London time)
    # Morning: 7:45 to 9:45, Afternoon: 14:45 to 16:45
    ts = pd.to_datetime(time, unit='s', utc=True)
    
    # London trading windows
    london_morning_start = ts.dt.tz_localize(None).dt.floor('D') + pd.Timedelta(hours=7, minutes=45)
    london_morning_end = ts.dt.tz_localize(None).dt.floor('D') + pd.Timedelta(hours=9, minutes=45)
    london_afternoon_start = ts.dt.tz_localize(None).dt.floor('D') + pd.Timedelta(hours=14, minutes=45)
    london_afternoon_end = ts.dt.tz_localize(None).dt.floor('D') + pd.Timedelta(hours=16, minutes=45)
    
    # Adjust for London timezone offset
    london_offset = pd.Series([370 for _ in range(len(ts))], index=ts.index)  # 3:00 EST offset from UTC
    london_ts = ts - london_offset
    
    in_morning = (london_ts >= london_morning_start) & (london_ts < london_morning_end)
    in_afternoon = (london_ts >= london_afternoon_start) & (london_ts < london_afternoon_end)
    in_trading_window = in_morning | in_afternoon
    
    # Skip bars where indicators are NaN
    valid_bars = ~(bfvg.isna() | sfvg.isna() | obUp.isna() | obDown.isna() | volfilt.isna() | atrfilt.isna() | locfiltb.isna() | locfilts.isna())
    valid_bars = valid_bars.fillna(False)
    
    # Build entry conditions
    long_entry = bfvg & obUp & in_trading_window & valid_bars
    short_entry = sfvg & obDown & in_trading_window & valid_bars
    
    # Iterate through bars to find entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_entry.iloc[i]:
            entry_ts = int(time.iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            entry_ts = int(time.iloc[i])
            entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time_str,
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
    
    return entries