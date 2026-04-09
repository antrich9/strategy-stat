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
    
    # Wilder RSI implementation
    def wilders_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilder ATR implementation
    def wilders_atr(df, period):
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    # Input parameters from Pine Script
    ext11 = 10
    upcolor11 = '#2962ff'
    downcolor11 = '#e91e63'
    inp1_11 = False  # Volume Filter
    inp2_11 = False  # ATR Filter
    inp3_11 = False  # Trend Filter
    FVGBKWY_on11 = True
    FVGImb11 = True
    FVGnew11 = True
    atrMultiplier11 = 3.0
    riskPerTrade11 = 1.0
    
    atr1 = wilders_atr(df, 14)
    atr211 = wilders_atr(df, 20) / 1.5
    
    # Build daily data manually from intraday data
    df_copy = df.copy()
    df_copy['datetime'] = pd.to_datetime(df_copy['time'], unit='s', utc=True)
    df_copy['date'] = df_copy['datetime'].dt.date
    
    daily_agg = df_copy.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'open': 'first',
        'time': 'last'
    })
    daily_agg = daily_agg.reset_index(drop=True)
    
    # Daily High/Low for current and previous days
    dailyHigh11 = daily_agg['high']
    dailyLow11 = daily_agg['low']
    dailyClose11 = daily_agg['close']
    dailyOpen11 = daily_agg['open']
    
    # Previous day's data
    prevDayHigh11 = dailyHigh11.shift(1)
    prevDayLow11 = dailyLow11.shift(1)
    
    # 1-day and 2-day shifted daily data
    dailyHigh21 = dailyHigh11.shift(1)
    dailyLow21 = dailyLow11.shift(1)
    dailyHigh22 = dailyHigh11.shift(2)
    dailyLow22 = dailyLow11.shift(2)
    
    # Volume filter
    volfilt11_series = df['volume'].shift(1) > df['volume'].shift(1).rolling(9).mean() * 1.5
    volfilt11 = inp1_11 if inp1_11 else volfilt11_series
    
    # ATR Filter
    atrfilt11 = inp2_11 if inp2_11 else ((dailyLow22 - dailyHigh22 > atr211) | (dailyLow22 - dailyHigh11 > atr211))
    
    # Trend Filter (LOC)
    loc11 = df['close'].rolling(54).mean()
    loc211 = loc11 > loc11.shift(1)
    locfiltb11 = loc211 if inp3_11 else pd.Series(True, index=df.index)
    locfilts11 = ~loc211 if inp3_11 else pd.Series(True, index=df.index)
    
    # FVG conditions
    bfvg11 = (dailyLow11 > dailyHigh22) & volfilt11 & atrfilt11 & locfiltb11
    sfvg11 = (dailyHigh11 < dailyLow22) & volfilt11 & atrfilt11 & locfilts11
    
    # Swing detection function
    is_swing_high11 = (dailyHigh21 < dailyHigh22) & (dailyHigh11.shift(3) < dailyHigh22) & (dailyHigh11.shift(4) < dailyHigh22)
    is_swing_low11 = (dailyLow21 > dailyLow22) & (dailyLow11.shift(3) > dailyLow22) & (dailyLow11.shift(4) > dailyLow22)
    
    # Track last swing type
    lastSwingType11 = pd.Series('none', index=daily_agg.index)
    
    # Build boolean series for each entry condition
    long_condition = bfvg11 & (lastSwingType11 == 'dailyLow')
    short_condition = sfvg11 & (lastSwingType11 == 'dailyHigh')
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        ts = int(df['time'].iloc[i])
        entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        # Check long entry
        if long_condition.iloc[i] and not pd.isna(df['close'].iloc[i]):
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        
        # Check short entry
        if short_condition.iloc[i] and not pd.isna(df['close'].iloc[i]):
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries