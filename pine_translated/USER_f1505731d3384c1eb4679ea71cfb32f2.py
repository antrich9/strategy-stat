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
    # Ensure we have enough data
    if len(df) < 5:
        return []

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

    # Extract hour from timestamp
    hours = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).hour)
    
    # Time filter: (hour >= 2 and hour < 5) or (hour >= 10 and hour < 12)
    is_valid_time = ((hours >= 2) & (hours < 5)) | ((hours >= 10) & (hours < 12))
    is_valid_time_series = is_valid_time.astype(bool)

    # OB and FVG conditions
    ob_up = pd.Series([False] * len(df), index=df.index)
    ob_down = pd.Series([False] * len(df), index=df.index)
    fvg_up = pd.Series([False] * len(df), index=df.index)
    fvg_down = pd.Series([False] * len(df), index=df.index)

    for i in range(1, len(df) - 2):
        try:
            ob_up.iloc[i] = is_ob_up(1) if i > 0 else False
            ob_down.iloc[i] = is_ob_down(1) if i > 0 else False
            fvg_up.iloc[i] = is_fvg_up(0) if i < len(df) - 2 else False
            fvg_down.iloc[i] = is_fvg_down(0) if i < len(df) - 2 else False
        except:
            pass

    # Stacked OB+FVG conditions
    stacked_bullish = ob_up & fvg_up
    stacked_bearish = ob_down & fvg_down

    # Wilder RSI implementation (length 14)
    def wilder_rsi(series, length=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Wilder ATR implementation (length 14)
    def wilder_atr(high, low, close, length=14):
        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/length, adjust=False).mean()
        return atr

    # Calculate indicators
    sma_volume = df['volume'].rolling(9).mean()
    vol_filt = sma_volume * 1.5
    volume_cond = df['volume'].shift(1) > vol_filt

    atr_20 = wilder_atr(df['high'], df['low'], df['close'], 20)
    atr_value = atr_20 / 1.5
    atr_filt_up = df['low'] - df['high'].shift(2) > atr_value
    atr_filt_down = df['low'].shift(2) - df['high'] > atr_value
    atr_filt = atr_filt_up | atr_filt_down

    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    loc_filt_bull = loc2
    loc_filt_bear = ~loc2

    # FVG conditions with filters
    bfvg = (df['low'] > df['high'].shift(2)) & volume_cond & atr_filt & loc_filt_bull
    sfvg = (df['high'] < df['low'].shift(2)) & volume_cond & atr_filt & loc_filt_bear

    # Previous day high/low (using shift for security-like behavior)
    prev_day_high = df['high'].shift(1).rolling('1D').max()
    prev_day_low = df['low'].shift(1).rolling('1D').min()
    current_day_high = df['high'].rolling('240T').max()
    current_day_low = df['low'].rolling('240T').min()

    # PDHL Sweep conditions
    bullish_sweep = (df['high'] >= prev_day_high) & (df['close'] < prev_day_high)
    bearish_sweep = (df['low'] <= prev_day_low) & (df['close'] > prev_day_low)

    # Combined entry conditions
    long_condition = is_valid_time_series & (stacked_bullish | bfvg) & bullish_sweep
    short_condition = is_valid_time_series & (stacked_bearish | sfvg) & bearish_sweep

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(df['close'].iloc[i]):
            continue
        
        entry_price = df['close'].iloc[i]
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        if long_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
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