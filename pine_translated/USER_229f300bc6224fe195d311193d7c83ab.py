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
    n = len(df)
    entries = []
    trade_num = 1

    # Helper functions for OB/FVG detection
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

    # Time filter
    def is_valid_trade_time(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        return (hour >= 2 and hour < 5) or (hour >= 10 and hour < 12)

    # Calculate filters
    # Volume filter: volume[1] > sma(volume, 9) * 1.5
    vol_sma = df['volume'].rolling(9).mean()
    vol_filter = df['volume'].shift(1) > vol_sma * 1.5

    # ATR filter: ta.atr(20) / 1.5
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean() / 1.5
    atrfilt = (low - high.shift(2) > atr) | (low.shift(2) - high > atr)

    # Trend filter
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    # Precompute OB/FVG conditions
    ob_up = pd.Series([np.nan] * n, index=df.index)
    ob_down = pd.Series([np.nan] * n, index=df.index)
    fvg_up = pd.Series([np.nan] * n, index=df.index)
    fvg_down = pd.Series([np.nan] * n, index=df.index)

    for i in range(2, n):
        if i + 1 < n:
            ob_up.iloc[i] = 1.0 if is_ob_up(i - 1) else 0.0
            ob_down.iloc[i] = 1.0 if is_ob_down(i - 1) else 0.0
        if i - 2 >= 0:
            fvg_up.iloc[i] = 1.0 if is_fvg_up(i - 2) else 0.0
            fvg_down.iloc[i] = 1.0 if is_fvg_down(i - 2) else 0.0

    # Bullish conditions: bfvg = low > high[2] and volfilt and atrfilt and locfiltb
    bfvg = (df['low'] > df['high'].shift(2)) & vol_filter & atrfilt & locfiltb

    # Bearish conditions: sfvg = high < low[2] and volfilt and atrfilt and locfilts
    sfvg = (df['high'] < df['low'].shift(2)) & vol_filter & atrfilt & locfilts

    # Top imbalances
    top_imb_bway = (df['low'].shift(2) <= df['open'].shift(1)) & (df['high'] >= df['close'].shift(1)) & (df['close'] < df['low'].shift(1))
    top_imb_xbway = (df['low'].shift(2) <= df['open'].shift(1)) & (df['high'] >= df['close'].shift(1)) & (df['close'] > df['low'].shift(1))

    # Bottom imbalances
    bot_imb_bway = (df['high'].shift(2) >= df['open'].shift(1)) & (df['low'] <= df['close'].shift(1)) & (df['close'] > df['high'].shift(1))
    bot_imb_xbway = (df['high'].shift(2) >= df['open'].shift(1)) & (df['low'] <= df['close'].shift(1)) & (df['close'] < df['high'].shift(1))

    # Skip bars where indicators are NaN (need at least 3 bars of history)
    min_idx = 3

    for i in range(min_idx, n):
        # Check for NaN in key indicators
        if pd.isna(ob_up.iloc[i]) or pd.isna(fvg_up.iloc[i]):
            continue

        ts = int(df['time'].iloc[i])
        entry_price = float(df['close'].iloc[i])

        # Bullish entry: valid time AND (bfvg OR top_imb_bway) AND ob_up present
        long_cond = (is_valid_trade_time(ts) and 
                     ((bfvg.iloc[i] if not pd.isna(bfvg.iloc[i]) else False) or 
                      (top_imb_bway.iloc[i] if not pd.isna(top_imb_bway.iloc[i]) else False)) and
                     (ob_up.iloc[i] == 1.0 if not pd.isna(ob_up.iloc[i]) else False))

        # Bearish entry: valid time AND (sfvg OR bot_imb_bway) AND ob_down present
        short_cond = (is_valid_trade_time(ts) and 
                      ((sfvg.iloc[i] if not pd.isna(sfvg.iloc[i]) else False) or 
                       (bot_imb_bway.iloc[i] if not pd.isna(bot_imb_bway.iloc[i]) else False)) and
                      (ob_down.iloc[i] == 1.0 if not pd.isna(ob_down.iloc[i]) else False))

        if long_cond:
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        if short_cond:
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries