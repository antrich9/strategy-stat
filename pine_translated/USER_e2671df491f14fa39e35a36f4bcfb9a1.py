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
    
    def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        return atr
    
    # Parameters from Pine Script
    SwingPeriod = 50
    ATR_Coefficient = 1.0
    
    # Calculate indicators
    atr = wilder_atr(df['high'], df['low'], df['close'], 200)
    rsi = wilder_rsi(df['close'], 14)
    
    # HTF Level Detector variables (arrays to track pivots)
    MSH_P = []
    MSH_I = []
    MSL_P = []
    MSL_I = []
    
    # MSS Trigger flags
    MSS_H = False
    MSS_L = False
    
    entries = []
    trade_num = 1
    
    # Iterate through bars
    for i in range(len(df)):
        current_time = int(df['time'].iloc[i])
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        current_close = df['close'].iloc[i]
        current_open = df['open'].iloc[i]
        
        # Check for pivot high (ta.pivothigh)
        is_pivot_high = True
        for j in range(1, SwingPeriod + 1):
            if i - j < 0:
                is_pivot_high = False
                break
            if current_high <= df['high'].iloc[i - j]:
                is_pivot_high = False
                break
        
        # Check for pivot low (ta.pivotlow)
        is_pivot_low = True
        for j in range(1, SwingPeriod + 1):
            if i - j < 0:
                is_pivot_low = False
                break
            if current_low >= df['low'].iloc[i - j]:
                is_pivot_low = False
                break
        
        # Add pivot to arrays with offset for lookback
        if is_pivot_high and i >= SwingPeriod:
            pivot_idx = i - SwingPeriod
            MSH_P.append(df['high'].iloc[pivot_idx])
            MSH_I.append(int(df['time'].iloc[pivot_idx]))
        
        if is_pivot_low and i >= SwingPeriod:
            pivot_idx = i - SwingPeriod
            MSL_P.append(df['low'].iloc[pivot_idx])
            MSL_I.append(int(df['time'].iloc[pivot_idx]))
        
        # Skip if not enough data for conditions
        if i < SwingPeriod:
            continue
        
        # Get current ATR value
        current_atr = atr.iloc[i]
        if pd.isna(current_atr) or current_atr == 0:
            continue
        
        # Check for FVG (Fair Value Gap) - 3-bar imbalance
        fvg_bullish = False
        fvg_bearish = False
        
        if i >= 2:
            prev_high = df['high'].iloc[i - 2]
            prev_low = df['low'].iloc[i - 2]
            mid_open = df['open'].iloc[i - 1]
            mid_close = df['close'].iloc[i - 1]
            curr_high = df['high'].iloc[i]
            curr_low = df['low'].iloc[i]
            
            # Bullish FVG: gap between bar i-2 high and bar i low
            if prev_high < curr_low:
                fvg_bullish = True
            
            # Bearish FVG: gap between bar i-2 low and bar i high
            if prev_low > curr_high:
                fvg_bearish = True
        
        # Check for HTF Level break (Major Swing High/Low break)
        htf_bull_break = False
        htf_bear_break = False
        
        if len(MSH_P) > 0 and not pd.isna(MSH_P[-1]):
            last_msh = MSH_P[-1]
            # Bullish break: close above MSH with ATR buffer
            if current_close > last_msh + (current_atr * ATR_Coefficient):
                htf_bull_break = True
                # Update MSS_H flag
                if not MSS_H:
                    MSS_H = True
        
        if len(MSL_P) > 0 and not pd.isna(MSL_P[-1]):
            last_msl = MSL_P[-1]
            # Bearish break: close below MSL with ATR buffer
            if current_close < last_msl - (current_atr * ATR_Coefficient):
                htf_bear_break = True
                # Update MSS_L flag
                if not MSS_L:
                    MSS_L = True
        
        # Detect MSS (Market Structure Shift) - change in polarity
        mss_bullish = False
        mss_bearish = False
        
        # Check if price breaks above last pivot high (swing failure for bulls)
        if len(MSH_P) >= 2 and not pd.isna(MSH_P[-1]) and not pd.isna(MSH_P[-2]):
            if current_close > MSH_P[-1] and current_close < MSH_P[-2]:
                mss_bullish = True
        
        # Check if price breaks below last pivot low (swing failure for bears)
        if len(MSL_P) >= 2 and not pd.isna(MSL_P[-1]) and not pd.isna(MSL_P[-2]):
            if current_close < MSL_P[-1] and current_close > MSL_P[-2]:
                mss_bearish = True
        
        # Entry conditions based on multiple confirmations
        long_entry = (htf_bull_break or mss_bullish) and (MSS_H or len(MSH_P) > 0)
        short_entry = (htf_bear_break or mss_bearish) and (MSS_L or len(MSL_P) > 0)
        
        # Alternative simpler logic: crossover/crossunder with last swing points
        if i >= 1:
            prev_close = df['close'].iloc[i - 1]
            
            # Check for crossover (long entry)
            if len(MSH_P) > 0 and not pd.isna(MSH_P[-1]):
                if prev_close <= MSH_P[-1] and current_close > MSH_P[-1]:
                    long_entry = True
            
            # Check for crossunder (short entry)
            if len(MSL_P) > 0 and not pd.isna(MSL_P[-1]):
                if prev_close >= MSL_P[-1] and current_close < MSL_P[-1]:
                    short_entry = True
        
        # Generate entry signals
        if long_entry:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': current_time,
                'entry_time': datetime.fromtimestamp(current_time, tz=timezone.utc).isoformat(),
                'entry_price_guess': current_close,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': current_close,
                'raw_price_b': current_close
            })
            trade_num += 1
            MSS_L = False  # Reset opposite flag
        
        if short_entry:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': current_time,
                'entry_time': datetime.fromtimestamp(current_time, tz=timezone.utc).isoformat(),
                'entry_price_guess': current_close,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': current_close,
                'raw_price_b': current_close
            })
            trade_num += 1
            MSS_H = False  # Reset opposite flag
    
    return entries