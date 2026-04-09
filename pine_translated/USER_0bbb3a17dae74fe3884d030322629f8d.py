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
    
    # London time windows (UTC offset for Europe/London is typically 0 or 1 depending on DST)
    # Using fixed offsets: morning 7:45-9:45 UTC, afternoon 14:45-16:45 UTC
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Europe/London')
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['dayofweek'] = df['datetime'].dt.dayofweek  # Monday=0, Friday=4
    
    # Morning window: 7:45 to 9:45 (hour 7 minute >= 45 OR hour 8 OR hour 9 minute < 45)
    is_morning = ((df['hour'] == 7) & (df['minute'] >= 45)) | \
                 ((df['hour'] == 8)) | \
                 ((df['hour'] == 9) & (df['minute'] < 45))
    
    # Afternoon window: 14:45 to 16:45
    is_afternoon = ((df['hour'] == 14) & (df['minute'] >= 45)) | \
                   ((df['hour'] == 15)) | \
                   ((df['hour'] == 16) & (df['minute'] < 45))
    
    is_within_time_window = is_morning | is_afternoon
    
    # Friday morning check
    is_friday_morning = (df['dayofweek'] == 4) & is_morning
    
    # Wilder RSI for length 14
    def wilder_rsi(series, length=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilder ATR
    def wilder_atr(high, low, close, length=14):
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        return atr
    
    # Calculate indicators
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Volume filter: volume[1] > sma(volume, 9) * 1.5
    vol_sma9 = volume.shift(1).rolling(9).mean()
    vol_filter = volume.shift(1) > vol_sma9 * 1.5
    
    # ATR filter (using 20 period)
    atr20 = wilder_atr(high, low, close, 20)
    atr_div = atr20 / 1.5
    # (low - high[2] > atr) or (low[2] - high > atr)
    atr_filter = (low - high.shift(2) > atr_div) | (low.shift(2) - high > atr_div)
    
    # Trend filter: SMA 54
    sma54 = close.rolling(54).mean()
    trend_up = sma54 > sma54.shift(1)
    trend_filter_long = trend_up
    trend_filter_short = ~trend_up
    
    # FVG detection
    # Bullish FVG: low > high[2]
    bull_fvg = low > high.shift(2)
    # Bearish FVG: high < low[2]
    bear_fvg = high < low.shift(2)
    
    # Apply filters to FVG detection
    bfvg = bull_fvg & vol_filter & atr_filter & trend_filter_long
    sfvg = bear_fvg & vol_filter & atr_filter & trend_filter_short
    
    # Swing detection (for stop loss reference, not used in entry but needed for context)
    # is_swing_high: high[1] < high[2] and high[3] < high[2] and high[4] < high[2]
    # is_swing_low: low[1] > low[2] and low[3] > low[2] and low[4] > low[2]
    # These are used to track last_swing_high and last_swing_low
    is_swing_high = (high.shift(1) < high.shift(2)) & \
                     (high.shift(3) < high.shift(2)) & \
                     (high.shift(4) < high.shift(2))
    is_swing_low = (low.shift(1) > low.shift(2)) & \
                    (low.shift(3) > low.shift(2)) & \
                    (low.shift(4) > low.shift(2))
    
    # Track last swing high/low (shift by 2 because swing detection is based on bar[2])
    last_swing_high = high.shift(2) * is_swing_high
    last_swing_low = low.shift(2) * is_swing_low
    
    # Fill forward last swing high/low
    last_swing_high = last_swing_high.replace(0, np.nan).ffill()
    last_swing_low = last_swing_low.replace(0, np.nan).ffill()
    
    # Track last FVG type and entry type (need to track across bars)
    # We'll simulate this with shifted series
    last_fvg_bull = bfvg.shift(1).fillna(False).astype(int)
    last_fvg_bear = sfvg.shift(1).fillna(False).astype(int)
    
    # Track last_entry (FVG entry signal)
    bull_entry_signal = bfvg & last_fvg_bear.shift(1).astype(bool)
    bear_entry_signal = sfvg & last_fvg_bull.shift(1).astype(bool)
    
    # Final entry conditions
    # Long: bull_entry_signal and in_trading_window and not friday_morning
    # Short: bear_entry_signal and in_trading_window and not friday_morning
    
    long_entries = bull_entry_signal & is_within_time_window & ~is_friday_morning
    short_entries = bear_entry_signal & is_within_time_window & ~is_friday_morning
    
    entries = []
    trade_num = 1
    
    # Iterate through bars
    for i in range(len(df)):
        if long_entries.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_entries.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries