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
    
    # Volume filter: volume[1] > sma(volume, 9) * 1.5
    sma_vol = df['volume'].rolling(window=9).mean()
    volfilt = (df['volume'].shift(1) > sma_vol * 1.5).fillna(True)
    
    # Wilder ATR (14 period)
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean() / 1.5
    
    # ATR filter
    atrfilt = ((low - high.shift(2) > atr) | (low.shift(2) - high > atr)).fillna(True)
    
    # Trend filter (SMA 54)
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2.fillna(True)
    locfilts = (~loc2).fillna(True)
    
    # FVG detection
    bfvg = (low > high.shift(2)) & (volfilt) & (atrfilt) & (locfiltb)
    sfvg = (high < low.shift(2)) & (volfilt) & (atrfilt) & (locfilts)
    
    # Time window (Europe/London: 7:45-9:45 and 14:45-16:45)
    times = pd.to_datetime(df['time'], unit='s', utc=True)
    hours = times.dt.hour
    minutes = times.dt.minute
    time_decimal = hours + minutes / 60.0
    
    in_morning = (time_decimal >= 7.75) & (time_decimal < 9.75)
    in_afternoon = (time_decimal >= 14.75) & (time_decimal < 16.75)
    isWithinTimeWindow = in_morning | in_afternoon
    
    # Swing detection (requires 4 bars lookback)
    # Swing High: high[1] < high[2] AND high[3] < high[2] AND high[4] < high[2]
    # Swing Low: low[1] > low[2] AND low[3] > low[2] AND low[4] > low[2]
    is_swing_high = (high.shift(1) < high.shift(2)) & \
                    (high.shift(3) < high.shift(2)) & \
                    (high.shift(4) < high.shift(2))
    is_swing_low = (low.shift(1) > low.shift(2)) & \
                   (low.shift(3) > low.shift(2)) & \
                   (low.shift(4) > low.shift(2))
    
    # Storing last swing high/low
    last_swing_high = pd.Series(np.nan, index=df.index)
    last_swing_low = pd.Series(np.nan, index=df.index)
    
    last_sh = np.nan
    last_sl = np.nan
    
    for i in range(5, n):
        if is_swing_high.iloc[i-1]:
            last_sh = high.iloc[i-2]
        if is_swing_low.iloc[i-1]:
            last_sl = low.iloc[i-2]
        last_swing_high.iloc[i] = last_sh
        last_swing_low.iloc[i] = last_sl
    
    # OB detection (isObUp, isObDown)
    close_gt_open = close > open
    close_lt_open = close < open
    
    # isObUp(index): isDown(index+1) and isUp(index) and close[index] > high[index+1]
    obUp = close_lt_open.shift(-1) & close_gt_open & (close > high.shift(-1))
    # isObDown(index): isUp(index+1) and isDown(index) and close[index] < low[index+1]
    obDown = close_gt_open.shift(-1) & close_lt_open & (close < low.shift(-1))
    
    # fvgUp/Down
    fvgUp = low > high.shift(2)
    fvgDown = high < low.shift(2)
    
    # Stacked OB+FVG (current bar is FVG, previous bar is OB)
    obUp_prev = obUp.shift(1).fillna(False)
    obDown_prev = obDown.shift(1).fillna(False)
    fvgUp_curr = fvgUp.fillna(False)
    fvgDown_curr = fvgDown.fillna(False)
    
    stacked_bullish = obUp_prev & fvgUp_curr
    stacked_bearish = obDown_prev & fvgDown_curr
    
    # Entry conditions
    # Long: bfvg AND time window
    # Short: sfvg AND time window
    long_cond = bfvg.fillna(False) & isWithinTimeWindow.fillna(False)
    short_cond = sfvg.fillna(False) & isWithinTimeWindow.fillna(False)
    
    entries = []
    trade_num = 1
    
    for i in range(n):
        # Need at least 5 bars for swing detection
        if i < 5:
            continue
        
        entry_price = close.iloc[i]
        
        if long_cond.iloc[i]:
            entry_time = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
        
        if short_cond.iloc[i]:
            entry_time = datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
    
    return entries