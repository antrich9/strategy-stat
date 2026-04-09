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
    # Make a working copy to avoid modifying original
    data = df.copy().reset_index(drop=True)
    
    # Helper: Wilder RSI
    def wilder_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Helper: Wilder ATR
    def wilder_atr(high, low, close, period):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    # Calculate EMA 50 and EMA 200
    ema_50 = data['close'].ewm(span=50, adjust=False).mean()
    ema_200 = data['close'].ewm(span=200, adjust=False).mean()
    
    # Calculate RSI(14)
    rsi = wilder_rsi(data['close'], 14)
    
    # Calculate ATR(20)
    atr = wilder_atr(data['high'], data['low'], data['close'], 20)
    
    # Calculate volume SMA(9)
    vol_sma_9 = data['volume'].rolling(9).mean()
    
    # Calculate SMA(54) for trend filter
    sma_54 = data['close'].rolling(54).mean()
    
    # Swing detection
    # main_bar_high = high[2], main_bar_low = low[2]
    # Swing High: high[1] < main_bar_high and high[3] < main_bar_high and high[4] < main_bar_high
    # Swing Low: low[1] > main_bar_low and low[3] > main_bar_low and low[4] > main_bar_low
    main_bar_high = data['high'].shift(2)
    main_bar_low = data['low'].shift(2)
    
    is_swing_high = (data['high'].shift(1) < main_bar_high) & (data['high'].shift(3) < main_bar_high) & (data['high'].shift(4) < main_bar_high)
    is_swing_low = (data['low'].shift(1) > main_bar_low) & (data['low'].shift(3) > main_bar_low) & (data['low'].shift(4) > main_bar_low)
    
    # Initialize last_swing_high and last_swing_low
    last_swing_high = pd.Series(np.nan, index=data.index)
    last_swing_low = pd.Series(np.nan, index=data.index)
    
    # Track last FVG type: 1 = Bullish, -1 = Bearish, 0 = None
    last_fvg = 0
    
    # RSI threshold
    rsi_threshold = 50
    
    # FVG conditions
    # bullish FVG: low > high[2]
    # bearish FVG: high < low[2]
    bfvg = data['low'] > data['high'].shift(2)
    sfvg = data['high'] < data['low'].shift(2)
    
    # Trend filter for bullish (loc2 > loc[1])
    loc2 = sma_54 > sma_54.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # ATR filter: (low - high[2] > atr/1.5) or (low[2] - high > atr/1.5)
    # Note: original uses atr/1.5 which is atr value / 1.5
    atr_threshold = atr / 1.5
    atrfilt_bull = (data['low'] - data['high'].shift(2)) > atr_threshold
    atrfilt_bear = (data['low'].shift(2) - data['high']) > atr_threshold
    atrfilt = atrfilt_bull | atrfilt_bear
    
    # Volume filter: volume[1] > sma(volume, 9) * 1.5
    volfilt = data['volume'].shift(1) > vol_sma_9 * 1.5
    
    # Full FVG conditions with filters
    bull_fvg_full = bfvg & locfiltb & atrfilt
    bear_fvg_full = sfvg & locfilts & atrfilt
    
    entries = []
    trade_num = 0
    
    for i in range(1, len(data)):
        # Update last_swing_high and last_swing_low
        if is_swing_high.iloc[i]:
            last_swing_high.iloc[i] = main_bar_high.iloc[i]
        else:
            last_swing_high.iloc[i] = last_swing_high.iloc[i-1] if i > 0 else np.nan
            
        if is_swing_low.iloc[i]:
            last_swing_low.iloc[i] = main_bar_low.iloc[i]
        else:
            last_swing_low.iloc[i] = last_swing_low.iloc[i-1] if i > 0 else np.nan
        
        # Update last_fvg based on FVG
        if bull_fvg_full.iloc[i]:
            last_fvg = 1
        elif bear_fvg_full.iloc[i]:
            last_fvg = -1
        
        # Check entry conditions
        # Long entry: bull_fvg_full and lastFVG == -1 and rsi > 50
        # Short entry: bear_fvg_full and lastFVG == 1 and rsi < 50
        
        entry_price = data['close'].iloc[i]
        ts = int(data['time'].iloc[i])
        
        if bull_fvg_full.iloc[i] and last_fvg == -1 and rsi.iloc[i] > rsi_threshold:
            # Long entry
            trade_num += 1
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
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
            # After entry, update last_fvg to 1
            last_fvg = 1
            
        elif bear_fvg_full.iloc[i] and last_fvg == 1 and rsi.iloc[i] < rsi_threshold:
            # Short entry
            trade_num += 1
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
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
            # After entry, update last_fvg to -1
            last_fvg = -1
    
    return entries