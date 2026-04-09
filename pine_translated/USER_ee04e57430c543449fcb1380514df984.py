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
    
    # Helper functions
    def is_up(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]
    
    def is_down(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]
    
    def is_ob_up(idx):
        return is_down(idx + 1) and is_up(idx) and df['close'].iloc[idx] > df['high'].iloc[idx + 1]
    
    def is_ob_down(idx):
        return is_up(idx + 1) and is_down(idx) and df['close'].iloc[idx] < df['low'].iloc[idx + 1]
    
    def is_fvg_up(idx):
        return df['low'].iloc[idx] > df['high'].iloc[idx + 2]
    
    def is_fvg_down(idx):
        return df['high'].iloc[idx] < df['low'].iloc[idx + 2]
    
    # Wilde's RSI implementation
    def wilders_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilde's ATR implementation
    def wilders_atr(df, period):
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    # Calculate indicators
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_col = df['open']
    
    # Volume filter
    volfilt = volume.shift(1) > volume.rolling(9).mean() * 1.5
    
    # ATR filter (ATR / 1.5)
    atr = wilders_atr(df, 20) / 1.5
    atrfilt_long = (low - high.shift(2) > atr) | (low.shift(2) - high > atr)
    atrfilt_short = (low - high.shift(2) > atr) | (low.shift(2) - high > atr)
    
    # Trend filter (SMA 54)
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2  # for long
    locfilts = ~loc2  # for short
    
    # FVG conditions
    bfvg = (low > high.shift(2)) & volfilt & atrfilt_long & locfiltb
    sfvg = (high < low.shift(2)) & volfilt & atrfilt_short & locfilts
    
    # OB conditions
    obUp = is_ob_up(1)
    obDown = is_ob_down(1)
    fvgUp = is_fvg_up(0)
    fvgDown = is_fvg_down(0)
    
    # Time windows (London time 7:45-9:45 and 14:45-16:45)
    df_times = pd.to_datetime(df['time'], unit='ms', utc=True)
    hours = df_times.dt.hour
    minutes = df_times.dt.minute
    
    # Morning window: 7:45 - 9:45
    in_morning = ((hours == 7) & (minutes >= 45)) | ((hours == 8)) | ((hours == 9) & (minutes <= 45))
    # Afternoon window: 14:45 - 16:45
    in_afternoon = ((hours == 14) & (minutes >= 45)) | ((hours == 15)) | ((hours == 16) & (minutes <= 45))
    in_time_window = in_morning | in_afternoon
    
    # Stacked OB + FVG conditions (for long and short)
    stacked_bullish = obUp & fvgUp
    stacked_bearish = obDown & fvgDown
    
    # Entry conditions: FVG with stacked OB confirmation within time window
    long_entry = bfvg & stacked_bullish & in_time_window
    short_entry = sfvg & stacked_bearish & in_time_window
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(close.iloc[i]) or pd.isna(low.iloc[i]) or pd.isna(high.iloc[i]):
            continue
            
        if long_entry.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entry_price = close.iloc[i]
            
            entries.append({
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
            
        elif short_entry.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entry_price = close.iloc[i]
            
            entries.append({
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
    
    return entries