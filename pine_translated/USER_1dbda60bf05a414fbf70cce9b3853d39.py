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
    
    # Wilder ATR implementation
    def wilder_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    # Wilder RSI implementation
    def wilder_rsi(close, period=14):
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Parameters from Pine Script
    PP = 5  # Pivot Period
    atrLength = 14
    atrMultiplier = 1.5
    
    # Calculate ATR
    atr = wilder_atr(df['high'], df['low'], df['close'], atrLength)
    
    # Calculate RSI
    rsi = wilder_rsi(df['close'], 14)
    
    # Detect pivot highs and lows
    pivot_high = df['high'].rolling(window=PP+1, center=True).max() == df['high']
    pivot_low = df['low'].rolling(window=PP+1, center=True).min() == df['low']
    
    # Mark pivot indices
    pivot_high_idx = pivot_high.astype(int)
    pivot_low_idx = pivot_low.astype(int)
    
    # Track last pivot values
    last_pivot_high = np.nan
    last_pivot_low = np.nan
    last_pivot_high_idx = -1
    last_pivot_low_idx = -1
    
    # Structure tracking
    bullish_choch = False
    bearish_choch = False
    bullish_bos = False
    bearish_bos = False
    
    # Track if we're in a structure (for ChoCh detection)
    in_uptrend = False
    in_downtrend = False
    
    # Entry flags from Pine Script
    dtTradeTriggered = False  # Bullish entry
    dbTradeTriggered = False  # Bearish entry
    
    # Track open positions
    isLongOpen = False
    isShortOpen = False
    
    entries = []
    trade_num = 1
    
    for i in range(PP + 1, len(df)):
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        current_close = df['close'].iloc[i]
        current_time = df['time'].iloc[i]
        current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0
        
        # Update pivot tracking
        if pivot_high_idx.iloc[i] == 1:
            last_pivot_high = current_high
            last_pivot_high_idx = i
            
        if pivot_low_idx.iloc[i] == 1:
            last_pivot_low = current_low
            last_pivot_low_idx = i
        
        # Detect structure: Higher High, Higher Low (uptrend)
        # Lower High, Lower Low (downtrend)
        if last_pivot_high_idx > 0 and last_pivot_low_idx > 0:
            prev_high_idx = max(0, last_pivot_high_idx - 1)
            prev_low_idx = max(0, last_pivot_low_idx - 1)
            
            if prev_high_idx < len(df) and prev_low_idx < len(df):
                prev_pivot_high = df['high'].iloc[prev_high_idx]
                prev_pivot_low = df['low'].iloc[prev_low_idx]
                
                # Higher High and Higher Low = Bullish
                is_hh = last_pivot_high > prev_pivot_high
                is_hl = last_pivot_low > prev_pivot_low
                
                # Lower High and Lower Low = Bearish
                is_lh = last_pivot_high < prev_pivot_high
                is_ll = last_pivot_low < prev_pivot_low
                
                # Detect BoS (Break of Structure)
                if is_hh and is_hl:
                    bullish_bos = True
                    bearish_bos = False
                    bullish_choch = False
                    bearish_choch = False
                elif is_lh and is_ll:
                    bearish_bos = True
                    bullish_bos = False
                    bullish_choch = False
                    bearish_choch = False
                
                # Detect ChoCh (Change of Character)
                if in_uptrend and (is_lh or is_ll):
                    bullish_choch = True
                    bearish_choch = False
                    in_uptrend = False
                    in_downtrend = True
                elif in_downtrend and (is_hh or is_hl):
                    bearish_choch = True
                    bullish_choch = False
                    in_downtrend = False
                    in_uptrend = True
                
                # Update trend state
                if is_hh and is_hl:
                    in_uptrend = True
                    in_downtrend = False
                elif is_lh and is_ll:
                    in_downtrend = True
                    in_uptrend = False
        
        # Entry conditions based on structure and RSI
        # DT (Detrend) Long Entry: Bullish structure + RSI confirmation
        dt_long_condition = (bullish_bos or bullish_choch) and not isLongOpen
        
        # DB (Directional Bias) Short Entry: Bearish structure + RSI confirmation
        db_short_condition = (bearish_bos or bearish_choch) and not isShortOpen
        
        # Additional filter: price near recent low for longs, high for shorts
        lookback_low = df['low'].iloc[max(0, i-PP):i].min()
        lookback_high = df['high'].iloc[max(0, i-PP):i].max()
        
        # Long entry: price pulls back to recent low in uptrend
        pullback_long = current_close <= lookback_low * 1.02 if lookback_low > 0 else False
        
        # Short entry: price pulls back to recent high in downtrend
        pullback_short = current_close >= lookback_high * 0.98 if lookback_high > 0 else False
        
        # Set entry triggers
        dtTradeTriggered = dt_long_condition and pullback_long
        dbTradeTriggered = db_short_condition and pullback_short
        
        # Execute entries
        if dtTradeTriggered and not isLongOpen:
            entry_price = current_close
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(current_time),
                'entry_time': datetime.fromtimestamp(current_time, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            isLongOpen = True
            dtTradeTriggered = False
        
        if dbTradeTriggered and not isShortOpen:
            entry_price = current_close
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(current_time),
                'entry_time': datetime.fromtimestamp(current_time, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            isShortOpen = True
            dbTradeTriggered = False
        
        # Reset entry flags on next bar (for next iteration)
        dtTradeTriggered = False
        dbTradeTriggered = False
    
    return entries