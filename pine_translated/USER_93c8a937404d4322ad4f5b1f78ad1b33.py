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
    
    # Helper functions for indicators
    def wilder_atr(high, low, close, period):
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    def wilder_rsi(close, period):
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate ATR
    atr14 = wilder_atr(df['high'], df['low'], df['close'], 14)
    atr20 = wilder_atr(df['high'], df['low'], df['close'], 20)
    atr21 = atr20 / 1.5
    
    # Calculate volume filter
    vol_sma = df['volume'].rolling(9).mean()
    volfilt = df['volume'].shift(1) > vol_sma * 1.5
    
    # Calculate ATR filter
    atrfilt = ((df['low'].shift(2) - df['high'].shift(2)) > atr21) | ((df['low'].shift(2) - df['low'].shift(1)) > atr21)
    
    # Calculate location/trend filter
    loc1 = df['close'].rolling(54).mean()
    loc2 = loc1 > loc1.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # Calculate daily PDH and PDL for bias detection
    df['date'] = pd.to_datetime(df['time'], unit='s').dt.date
    daily_agg = df.groupby('date').agg({'high': 'max', 'low': 'min'}).reset_index()
    daily_agg.columns = ['date', 'daily_high', 'daily_low']
    daily_agg['prev_daily_high'] = daily_agg['daily_high'].shift(1)
    daily_agg['prev_daily_low'] = daily_agg['daily_low'].shift(1)
    df = df.merge(daily_agg[['date', 'prev_daily_high', 'prev_daily_low']], on='date', how='left')
    
    # Calculate bias
    sweptLow = df['low'] < df['prev_daily_low']
    sweptHigh = df['high'] > df['prev_daily_high']
    brokeHigh = df['close'] > df['prev_daily_high']
    brokeLow = df['close'] < df['prev_daily_low']
    
    # Detect new day
    newDay = pd.Series([False] * len(df))
    newDay.iloc[1:] = df['date'].iloc[1:].values != df['date'].iloc[:-1].values
    
    # Initialize bias series
    bias = pd.Series([0] * len(df), index=df.index)
    
    # Calculate bias with forward fill
    for i in range(1, len(df)):
        if newDay.iloc[i]:
            bias.iloc[i] = 0
        elif sweptLow.iloc[i] and brokeHigh.iloc[i]:
            bias.iloc[i] = 1
        elif sweptHigh.iloc[i] and brokeLow.iloc[i]:
            bias.iloc[i] = -1
        elif df['low'].iloc[i] < df['prev_daily_low'].iloc[i]:
            bias.iloc[i] = -1
        elif df['high'].iloc[i] > df['prev_daily_high'].iloc[i]:
            bias.iloc[i] = 1
        else:
            bias.iloc[i] = bias.iloc[i-1]
    
    # Swing detection using 4H data
    # For 4H candles: shift references match Pine Script's dailyHigh21, dailyHigh22 logic
    swing_high = (df['high'].shift(2) < df['high'].shift(3)) & (df['high'].shift(1) < df['high'].shift(3)) & (df['high'] < df['high'].shift(3))
    swing_low = (df['low'].shift(2) > df['low'].shift(3)) & (df['low'].shift(1) > df['low'].shift(3)) & (df['low'] > df['low'].shift(3))
    
    # Track last swing type
    lastSwingType = pd.Series(["none"] * len(df), index=df.index)
    lastSwingType.iloc[1:] = np.where(swing_high.shift(1).iloc[1:], "dailyHigh",
                             np.where(swing_low.shift(1).iloc[1:], "dailyLow", "none"))
    lastSwingType = lastSwingType.replace("none", np.nan).ffill().fillna("none")
    
    # FVG detection
    bfvg = (df['low'].shift(1) > df['high'].shift(3)) & volfilt & atrfilt & locfiltb
    sfvg = (df['high'].shift(1) < df['low'].shift(3)) & volfilt & atrfilt & locfilts
    
    # Entry conditions with bias
    bull_entry_cond = bfvg & (lastSwingType == "dailyLow") & (bias == 1)
    bear_entry_cond = sfvg & (lastSwingType == "dailyHigh") & (bias == -1)
    
    # Build boolean series for entries
    bull_entry = bull_entry_cond & ~bull_entry_cond.shift(1).fillna(False)
    bear_entry = bear_entry_cond & ~bear_entry_cond.shift(1).fillna(False)
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(df['close'].iloc[i]):
            continue
        
        if bull_entry.iloc[i] if i < len(bull_entry) else False:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif bear_entry.iloc[i] if i < len(bear_entry) else False:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries