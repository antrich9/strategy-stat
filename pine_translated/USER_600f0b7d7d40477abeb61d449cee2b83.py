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
    
    # Session parameters
    session_start = "2300"
    session_end = "0700"
    timezone_str = "Europe/London"
    
    # Convert time to datetime for session detection
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert(timezone_str)
    df['hour'] = df['dt'].dt.hour
    
    # Parse session start and end hours
    start_hour = int(session_start[:2])
    start_min = int(session_start[2:])
    end_hour = int(session_end[:2])
    end_min = int(session_end[2:])
    
    # Determine if current time is in session
    def in_session(hour, minute):
        start_mins = start_hour * 60 + start_min
        end_mins = end_hour * 60 + end_min
        current_mins = hour * 60 + minute
        if start_mins < end_mins:
            return start_mins <= current_mins < end_mins
        else:
            return current_mins >= start_mins or current_mins < end_mins
    
    df['minute'] = df['dt'].dt.minute
    df['inSession'] = df.apply(lambda row: in_session(row['hour'], row['minute']), axis=1)
    df['inSession_prev'] = df['inSession'].shift(1).fillna(False)
    
    # Detect session start and end
    df['newSession'] = df['inSession'] & ~df['inSession_prev']
    df['sessionEnd'] = ~df['inSession'] & df['inSession_prev']
    
    # Calculate Asia session high and low
    asiaHigh = np.nan
    asiaLow = np.nan
    asiaHigh_arr = np.full(len(df), np.nan)
    asiaLow_arr = np.full(len(df), np.nan)
    asiaHighPlot_arr = np.full(len(df), np.nan)
    asiaLowPlot_arr = np.full(len(df), np.nan)
    
    for i in range(len(df)):
        if df['newSession'].iloc[i]:
            asiaHigh = df['high'].iloc[i]
            asiaLow = df['low'].iloc[i]
        elif df['inSession'].iloc[i]:
            if not np.isnan(asiaHigh):
                asiaHigh = max(asiaHigh, df['high'].iloc[i])
            if not np.isnan(asiaLow):
                asiaLow = min(asiaLow, df['low'].iloc[i])
        asiaHigh_arr[i] = asiaHigh
        asiaLow_arr[i] = asiaLow
        
        if df['sessionEnd'].iloc[i]:
            asiaHighPlot_arr[i] = asiaHigh
            asiaLowPlot_arr[i] = asiaLow
        elif not df['inSession'].iloc[i]:
            if i > 0:
                asiaHighPlot_arr[i] = asiaHighPlot_arr[i-1]
                asiaLowPlot_arr[i] = asiaLowPlot_arr[i-1]
    
    df['asiaHighPlot'] = asiaHighPlot_arr
    df['asiaLowPlot'] = asiaLowPlot_arr
    
    # Asia high/low sweep conditions
    df['asiahighSwept'] = df['high'] > df['asiaHighPlot']
    df['asialowSwept'] = df['low'] < df['asiaLowPlot']
    
    # OB and FVG detection functions
    def isUp(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]
    
    def isDown(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]
    
    def isObUp(idx):
        return isDown(idx + 1) and isUp(idx) and df['close'].iloc[idx] > df['high'].iloc[idx + 1]
    
    def isObDown(idx):
        return isUp(idx + 1) and isDown(idx) and df['close'].iloc[idx] < df['low'].iloc[idx + 1]
    
    def isFvgUp(idx):
        return df['low'].iloc[idx] > df['high'].iloc[idx + 2]
    
    def isFvgDown(idx):
        return df['high'].iloc[idx] < df['low'].iloc[idx + 2]
    
    # Calculate OB and FVG conditions (using previous bar index as in Pine script)
    df['obUp'] = False
    df['obDown'] = False
    df['fvgUp'] = False
    df['fvgDown'] = False
    
    for i in range(2, len(df)):
        try:
            df.loc[df.index[i], 'obUp'] = isObUp(i - 1)
            df.loc[df.index[i], 'obDown'] = isObDown(i - 1)
            df.loc[df.index[i], 'fvgUp'] = isFvgUp(i - 2)
            df.loc[df.index[i], 'fvgDown'] = isFvgDown(i - 2)
        except:
            pass
    
    # ATR calculation (Wilder ATR)
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    df['atr'] = atr / 1.5
    
    # Volume filter
    vol_sma = df['volume'].rolling(9).mean()
    df['volfilt'] = df['volume'].shift(1) > vol_sma * 1.5
    
    # ATR filter
    df['atrfilt'] = ((df['low'] - df['high'].shift(2) > df['atr']) | (df['low'].shift(2) - df['high'] > df['atr']))
    
    # Trend filter
    loc = df['close'].rolling(54).mean()
    loc_prev = loc.shift(1)
    df['locfiltb'] = loc > loc_prev
    df['locfilts'] = loc <= loc_prev
    
    # Bullish and Bearish FVG conditions
    df['bfvg'] = (df['low'] > df['high'].shift(2)) & df['volfilt'] & df['atrfilt'] & df['locfiltb']
    df['sfvg'] = (df['high'] < df['low'].shift(2)) & df['volfilt'] & df['atrfilt'] & df['locfilts']
    
    # Entry conditions
    # Long entry: Asia low swept AND (OB up OR FVG up) OR BFVG
    # Short entry: Asia high swept AND (OB down OR FVG down) OR SFVG
    df['long_condition'] = (df['asialowSwept'] & (df['obUp'] | df['fvgUp'])) | df['bfvg']
    df['short_condition'] = (df['asiahighSwept'] & (df['obDown'] | df['fvgDown'])) | df['sfvg']
    
    # Iterate through bars and generate entries
    for i in range(len(df)):
        if pd.isna(df['asiaHighPlot'].iloc[i]) or pd.isna(df['asiaLowPlot'].iloc[i]):
            continue
        
        # Check long entry
        if df['long_condition'].iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        # Check short entry
        if df['short_condition'].iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return results