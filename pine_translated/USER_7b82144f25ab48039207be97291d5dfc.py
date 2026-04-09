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
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    time = df['time']

    n = len(df)
    if n < 5:
        return []

    # Volume Filter
    vol_sma = volume.rolling(9).mean()
    volfilt = vol_sma * 1.5

    # ATR Filter (Wilder)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    atr_filt_val = atr / 1.5

    # Trend Filter (SMA based)
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)

    # EMA for trend filter
    short_ema = close.ewm(span=50, adjust=False).mean()
    long_ema = close.ewm(span=200, adjust=False).mean()
    trend_filter = short_ema > long_ema

    # Bullish and Bearish FVGs
    bfvg = (low > high.shift(2)) & volfilt.shift(1).fillna(False) & ((low - high.shift(2)) > atr_filt_val) & loc2
    sfvg = (high < low.shift(2)) & volfilt.shift(1).fillna(False) & ((low.shift(2) - high) > atr_filt_val) & (~loc2)

    # Swing detection (is_swing_high/is_swing_low 2 bars back)
    main_bar_high = high.shift(2)
    main_bar_low = low.shift(2)

    is_swing_high_cond = (high.shift(1) < main_bar_high) & (high.shift(3) < main_bar_high) & (high.shift(4) < main_bar_high)
    is_swing_low_cond = (low.shift(1) > main_bar_low) & (low.shift(3) > main_bar_low) & (low.shift(4) > main_bar_low)

    # State variables (var in Pine)
    lastFVG = 0  # 1=bullish, -1=bearish, 0=none
    currentSharpTurnBottom = np.nan
    currentSharpTurnTop = np.nan
    sharpTurnDirection = 0  # 1=bullish, -1=bearish
    sharpTurnBarIndex = 0

    entries = []
    trade_num = 1

    for i in range(5, n):
        # Update state based on current bar's FVG (bfvg[i], sfvg[i])
        # This affects next iteration's sharp turn detection

        # Check for sharp turn BEFORE updating lastFVG
        if bfvg.iloc[i] and lastFVG == -1:
            currentSharpTurnBottom = high.iloc[i-2]
            currentSharpTurnTop = low.iloc[i]
            sharpTurnDirection = 1
            sharpTurnBarIndex = i

        elif sfvg.iloc[i] and lastFVG == 1:
            currentSharpTurnTop = high.iloc[i]
            currentSharpTurnBottom = low.iloc[i-2]
            sharpTurnDirection = -1
            sharpTurnBarIndex = i

        # Update lastFVG
        if bfvg.iloc[i]:
            lastFVG = 1
        elif sfvg.iloc[i]:
            lastFVG = -1

        # Entry conditions
        # Long entry: sharpTurnDirection==1, i>sharpTurnBarIndex, crossunder(low, currentSharpTurnTop), trend_filter
        # Short entry: sharpTurnDirection==-1, i>sharpTurnBarIndex, crossover(high, currentSharpTurnBottom), not trend_filter

        long_entry = False
        short_entry = False

        if sharpTurnDirection == 1 and i > sharpTurnBarIndex:
            if not np.isnan(currentSharpTurnTop):
                # crossunder: low[i] < currentSharpTurnTop and low[i-1] >= currentSharpTurnTop
                if i > 0:
                    if low.iloc[i] < currentSharpTurnTop and low.iloc[i-1] >= currentSharpTurnTop:
                        if trend_filter.iloc[i]:
                            long_entry = True

        if sharpTurnDirection == -1 and i > sharpTurnBarIndex:
            if not np.isnan(currentSharpTurnBottom):
                # crossover: high[i] > currentSharpTurnBottom and high[i-1] <= currentSharpTurnBottom
                if i > 0:
                    if high.iloc[i] > currentSharpTurnBottom and high.iloc[i-1] <= currentSharpTurnBottom:
                        if not trend_filter.iloc[i]:
                            short_entry = True

        if long_entry:
            entry_ts = int(time.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])

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

            # Reset sharp turn after entry (like Pine: sharpTurnDirection := 0, etc.)
            sharpTurnDirection = 0
            currentSharpTurnTop = np.nan
            currentSharpTurnBottom = np.nan

        if short_entry:
            entry_ts = int(time.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])

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

            # Reset sharp turn after entry
            sharpTurnDirection = 0
            currentSharpTurnTop = np.nan
            currentSharpTurnBottom = np.nan

    return entries