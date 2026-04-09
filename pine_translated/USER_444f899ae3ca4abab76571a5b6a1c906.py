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
    
    # Convert time to datetime for time-based filtering
    df = df.copy()
    dt = pd.to_datetime(df['time'], unit='s', utc=True)
    df['dt'] = dt
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    df['dayofweek'] = df['dt'].dt.dayofweek
    
    # Time window: London morning (08:00-09:55) and afternoon (14:00-16:55)
    morning_window = ((df['hour'] == 8) | ((df['hour'] == 9) & (df['minute'] < 55)))
    afternoon_window = ((df['hour'] == 14) | ((df['hour'] >= 15) & (df['hour'] < 17)) | ((df['hour'] == 16) & (df['minute'] < 55)))
    in_time_window = morning_window | afternoon_window
    
    # Friday morning exclusion
    is_friday_morning = (df['dayofweek'] == 4) & morning_window
    
    # Volume filter: volume > SMA(volume, 9) * 1.5
    vol_sma = df['volume'].shift(1).rolling(9).mean()
    vol_filt = df['volume'].shift(1) > vol_sma * 1.5
    
    # ATR filter
    atr = ta_atr(df, 20)
    atr_filter = ((df['low'] - df['high'].shift(2) > atr / 1.5) | (df['low'].shift(2) - df['high'] > atr / 1.5))
    
    # Trend filter: SMA(close, 54) trending up or down
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    loc_filt_bull = loc2
    loc_filt_bear = ~loc2
    
    # Bullish FVG: low > high[2]
    bfvg = (df['low'] > df['high'].shift(2)) & vol_filt & atr_filter & loc_filt_bull
    
    # Bearish FVG: high < low[2]
    sfvg = (df['high'] < df['low'].shift(2)) & vol_filt & atr_filter & loc_filt_bear
    
    # Consecutive FVG counters
    consecutive_bfvg = np.zeros(len(df))
    consecutive_sfvg = np.zeros(len(df))
    flag_bfvg = False
    
    for i in range(1, len(df)):
        if bfvg.iloc[i]:
            consecutive_bfvg[i] = consecutive_bfvg[i-1] + 1
            consecutive_sfvg[i] = 0
            flag_bfvg = True
        elif sfvg.iloc[i]:
            consecutive_sfvg[i] = consecutive_sfvg[i-1] + 1
            consecutive_bfvg[i] = 0
            flag_bfvg = False
        else:
            consecutive_bfvg[i] = 0
            consecutive_sfvg[i] = 0
            flag_bfvg = False
    
    # Sharp turn detection: BFVG followed by CFVG (consecutiveSfvg >= 1 after consecutiveBfvg >= 1)
    sharp_turn_long = (consecutive_bfvg >= 1) & (np.roll(consecutive_sfvg, 1) >= 1)
    sharp_turn_short = (consecutive_sfvg >= 1) & (np.roll(consecutive_bfvg, 1) >= 1)
    sharp_turn_long[0] = False
    sharp_turn_short[0] = False
    
    # Combined entry conditions within trading window
    df['in_window'] = in_time_window & ~is_friday_morning
    df['bfvg'] = bfvg
    df['sfvg'] = sfvg
    
    # Entries: bullish/bearish FVG in trading window
    long_entry = bfvg & in_time_window & ~is_friday_morning
    short_entry = sfvg & in_time_window & ~is_friday_morning
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(df['close'].iloc[i]):
            continue
        if long_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
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
            trade_num += 1
        elif short_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
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
            trade_num += 1
    
    return entries

def ta_atr(df, length):
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/length, adjust=False).mean()
    
    return atr

def ta_sma(series, length):
    return series.rolling(length).mean()

def ta_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()