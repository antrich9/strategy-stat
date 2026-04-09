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
    entries = []
    trade_num = 1
    
    if len(df) < 3:
        return entries
    
    high = df['high']
    low = df['low']
    close = df['close']
    time_col = df['time']
    volume = df['volume']
    
    # Previous day high, low, close using daily rolling window
    daily_high = high.rolling('1D').max()
    daily_low = low.rolling('1D').min()
    daily_close = close.rolling('1D').last()
    daily_open = close.rolling('1D').first()
    
    # Detect new day
    day_change = time_col.diff().dt.days != 0
    new_day = day_change.fillna(False)
    
    # Previous day OHLC (shift by 1 to get previous day values)
    pdh = daily_high.shift(1)
    pdl = daily_low.shift(1)
    pdClose = daily_close.shift(1)
    
    # Current daily open
    dailyOpen = close.rolling('1D').first()
    
    # Is daily green or red
    isDailyGreen = close > dailyOpen
    isDailyRed = close < dailyOpen
    
    # Trade filters
    bullishAllowed = isDailyRed
    bearishAllowed = isDailyGreen
    
    # Asia session parameters (hardcoded defaults from Pine Script)
    asia_start_hour = 9
    asia_start_minute = 30
    asia_end_hour = 10
    asia_end_minute = 0
    
    # Convert timestamp to datetime for session detection
    dt = pd.to_datetime(time_col, unit='s', utc=True)
    sess_hour = dt.dt.hour
    sess_minute = dt.dt.minute
    sess_day = dt.dt.day
    
    # Session detection (simplified - assumes single day session)
    in_session = (sess_hour > asia_start_hour) | ((sess_hour == asia_start_hour) & (sess_minute >= asia_start_minute))
    in_session = in_session & ((sess_hour < asia_end_hour) | ((sess_hour == asia_end_hour) & (sess_minute < asia_end_minute)))
    
    # Session high/low tracking (var-like behavior)
    session_high = pd.Series(np.nan, index=df.index)
    session_low = pd.Series(np.nan, index=df.index)
    swept_high = pd.Series(False, index=df.index)
    swept_low = pd.Series(False, index=df.index)
    
    for i in range(1, len(df)):
        if in_session.iloc[i]:
            if np.isnan(session_high.iloc[i-1]):
                session_high.iloc[i] = high.iloc[i]
            else:
                session_high.iloc[i] = max(session_high.iloc[i-1], high.iloc[i])
            
            if np.isnan(session_low.iloc[i-1]):
                session_low.iloc[i] = low.iloc[i]
            else:
                session_low.iloc[i] = min(session_low.iloc[i-1], low.iloc[i])
        else:
            session_high.iloc[i] = session_high.iloc[i-1]
            session_low.iloc[i] = session_low.iloc[i-1]
        
        swept_high.iloc[i] = swept_high.iloc[i-1]
        swept_low.iloc[i] = swept_low.iloc[i-1]
    
    # Detect sweep conditions
    sweep_high_now = (~in_session) & (~swept_high) & (~pd.isna(session_high)) & (high > session_high)
    sweep_low_now = (~in_session) & (~swept_low) & (~pd.isna(session_low)) & (low < session_low)
    
    # 4H data aggregation for FVG detection
    dt_idx = pd.to_datetime(time_col, unit='s', utc=True)
    h4_high = high.resample('240min', offset='60min').max()
    h4_low = low.resample('240min').min()
    h4_close = close.resample('240min').last()
    h4_volume = volume.resample('240min').sum()
    
    # Reindex to match original df
    h4_high = h4_high.reindex(df.index, method='ffill')
    h4_low = h4_low.reindex(df.index, method='ffill')
    h4_close = h4_close.reindex(df.index, method='ffill')
    h4_volume = h4_volume.reindex(df.index, method='ffill')
    
    # Volume filter
    vol_sma = h4_volume.rolling(9).mean()
    volfilt = h4_volume.shift(1) > vol_sma.shift(1) * 1.5
    
    # ATR filter (manual Wilder ATR)
    tr = np.maximum(high - low, np.maximum(np.abs(high - h4_close.shift(1)), np.abs(low - h4_close.shift(1))))
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atrfilt = ((h4_low - h4_high.shift(2) > atr / 1.5) | (h4_low.shift(2) - h4_high > atr / 1.5))
    
    # Trend filter
    loc1 = h4_close.rolling(54).mean()
    locfiltb1 = loc1 > loc1.shift(1)
    locfilts1 = loc1 < loc1.shift(1)
    
    # FVG detection
    bfvg = (h4_low > h4_high.shift(2)) & volfilt & atrfilt & locfiltb1
    sfvg = (h4_high < h4_low.shift(2)) & volfilt & atrfilt & locfilts1
    
    # Track last FVG type
    lastFVG = pd.Series(0, index=df.index)
    
    for i in range(2, len(df)):
        if bfvg.iloc[i] and lastFVG.iloc[i-1] == -1:
            lastFVG.iloc[i] = 1
        elif sfvg.iloc[i] and lastFVG.iloc[i-1] == 1:
            lastFVG.iloc[i] = -1
        else:
            lastFVG.iloc[i] = lastFVG.iloc[i-1]
    
    # Sharp turn conditions
    bullSharpTurn = bfvg & (lastFVG.shift(1) == -1)
    bearSharpTurn = sfvg & (lastFVG.shift(1) == 1)
    
    # Entry conditions - bullish: bullishAllowed and bullSharpTurn
    # Entry conditions - bearish: bearishAllowed and bearSharpTurn
    long_entry = bullishAllowed & bullSharpTurn
    short_entry = bearishAllowed & bearSharpTurn
    
    # Build entries for long signals
    for i in range(len(df)):
        if i < 2:
            continue
        if pd.isna(pdh.iloc[i]) or pd.isna(pdl.iloc[i]):
            continue
        
        if long_entry.iloc[i]:
            ts = int(time_col.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            ts = int(time_col.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    return entries