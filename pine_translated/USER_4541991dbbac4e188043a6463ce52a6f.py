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
    
    # Parameters (from Pine Script inputs)
    atrLength = 14
    atrMultiplier = 1.5
    takeProfitRatio = 1.5
    tradeDirection = "Both"
    gmmaLength = 14
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    input_breakout = True
    input_retest = True
    rTon = True
    rTcc = False
    rThv = False
    
    bb = input_lookback
    
    # Initialize result list
    entries = []
    trade_num = 0
    
    # Helper functions for indicators
    def calculate_wilder_rsi(series, length):
        """Calculate Wilder's RSI"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_wilder_atr(df, length):
        """Calculate Wilder's ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Wilder's ATR
        atr = true_range.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        return atr
    
    # Calculate ATR
    atr = calculate_wilder_atr(df, atrLength)
    
    # Calculate pivot highs and lows
    def pivot_high(high, left_len, right_len):
        """Calculate pivot high"""
        result = pd.Series(np.nan, index=high.index)
        for i in range(right_len, len(high) - left_len):
            if all(high.iloc[i] >= high.iloc[i - left_len:i]) and all(high.iloc[i] >= high.iloc[i + 1:i + right_len + 1]):
                result.iloc[i] = high.iloc[i]
        return result
    
    def pivot_low(low, left_len, right_len):
        """Calculate pivot low"""
        result = pd.Series(np.nan, index=low.index)
        for i in range(right_len, len(low) - left_len):
            if all(low.iloc[i] <= low.iloc[i - left_len:i]) and all(low.iloc[i] <= low.iloc[i + 1:i + right_len + 1]):
                result.iloc[i] = low.iloc[i]
        return result
    
    ph = pivot_high(df['high'], bb, bb)
    pl = pivot_low(df['low'], bb, bb)
    
    # Forward fill NaN values (fixnan equivalent)
    pl = pl.ffill()
    ph = ph.ffill()
    
    # Initialize state variables
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)
    
    # Calculate box boundaries
    s_yLoc = pd.Series(np.nan, index=df.index)
    r_yLoc = pd.Series(np.nan, index=df.index)
    
    for i in range(bb + 1, len(df)):
        if df['low'].iloc[i - bb - 1] > df['low'].iloc[i - bb + 1]:
            s_yLoc.iloc[i] = df['low'].iloc[i - bb + 1]
        else:
            s_yLoc.iloc[i] = df['low'].iloc[i - bb - 1]
            
        if df['high'].iloc[i - bb - 1] > df['high'].iloc[i - bb + 1]:
            r_yLoc.iloc[i] = df['high'].iloc[i - bb - 1]
        else:
            r_yLoc.iloc[i] = df['high'].iloc[i - bb + 1]
    
    # Box boundaries
    sBot = s_yLoc.copy()
    rTop = r_yLoc.copy()
    sTop = pl.copy()
    rBot = ph.copy()
    
    # Calculate crossover and crossunder for breakouts
    def crossover(a, b):
        """Crossover: a crosses above b"""
        result = pd.Series(False, index=a.index)
        for i in range(1, len(a)):
            if a.iloc[i] > b.iloc[i] and a.iloc[i-1] <= b.iloc[i-1]:
                result.iloc[i] = True
        return result
    
    def crossunder(a, b):
        """Crossunder: a crosses below b"""
        result = pd.Series(False, index=a.index)
        for i in range(1, len(a)):
            if a.iloc[i] < b.iloc[i] and a.iloc[i-1] >= b.iloc[i-1]:
                result.iloc[i] = True
        return result
    
    # Repaint function
    def repaint(c1, c2, c3):
        if rTon:
            return c1
        elif rThv:
            return c2
        elif rTcc:
            return c3
        else:
            return pd.Series(False, index=c1.index)
    
    # Box change detection
    change_pl = pl != pl.shift(1)
    change_ph = ph != ph.shift(1)
    
    # Breakout conditions
    cu = repaint(
        crossunder(df['close'], sBot),
        crossunder(df['low'], sBot),
        crossunder(df['close'], sBot)
    )
    co = repaint(
        crossover(df['close'], rTop),
        crossover(df['high'], rTop),
        crossover(df['close'], rTop)
    )
    
    # Bars since function
    def barssince(condition):
        result = pd.Series(np.nan, index=condition.index)
        count = 0
        found = False
        for i in range(len(condition)):
            if pd.isna(result.iloc[i]):
                if condition.iloc[i]:
                    result.iloc[i] = count
                    found = True
                    count = 0
                else:
                    count += 1
        return result
    
    # Update breakout state
    for i in range(1, len(df)):
        if change_pl.iloc[i]:
            if pd.isna(sBreak.iloc[i-1]) or not sBreak.iloc[i-1]:
                sBreak.iloc[i] = False
            else:
                sBreak.iloc[i] = False
            if pd.isna(sBreak.iloc[i]):
                sBreak.iloc[i] = np.nan
        else:
            if cu.iloc[i] and pd.isna(sBreak.iloc[i-1]):
                sBreak.iloc[i] = True
            else:
                sBreak.iloc[i] = sBreak.iloc[i-1]
        
        if change_ph.iloc[i]:
            if pd.isna(rBreak.iloc[i-1]) or not rBreak.iloc[i-1]:
                rBreak.iloc[i] = False
            else:
                rBreak.iloc[i] = False
            if pd.isna(rBreak.iloc[i]):
                rBreak.iloc[i] = np.nan
        else:
            if co.iloc[i] and pd.isna(rBreak.iloc[i-1]):
                rBreak.iloc[i] = True
            else:
                rBreak.iloc[i] = rBreak.iloc[i-1]
    
    # Reset breakout on pivot change
    for i in range(len(df)):
        if change_pl.iloc[i]:
            sBreak.iloc[i] = np.nan
        if change_ph.iloc[i]:
            rBreak.iloc[i] = np.nan
    
    # Retest conditions
    def retest_condition(breakout_series, c1, c2, c3, c4):
        """Calculate retest conditions"""
        bars_since_breakout = barssince(~pd.isna(breakout_series) & (breakout_series == True))
        
        result = pd.Series(False, index=df.index)
        for i in range(len(df)):
            if bars_since_breakout.iloc[i] > input_retSince:
                if c1.iloc[i] or c2.iloc[i] or c3.iloc[i] or c4.iloc[i]:
                    result.iloc[i] = True
        return result
    
    # Support retest conditions
    s1 = retest_condition(sBreak, 
                          (df['high'] >= sTop) & (df['close'] <= sBot),
                          (df['high'] >= sTop) & (df['close'] >= sBot) & (df['close'] <= sTop),
                          (df['high'] >= sBot) & (df['high'] <= sTop),
                          (df['high'] >= sBot) & (df['high'] <= sTop) & (df['close'] < sBot))
    
    s2 = retest_condition(sBreak,
                          (df['high'] >= sTop) & (df['close'] >= sBot) & (df['close'] <= sTop),
                          (df['high'] >= sBot) & (df['high'] <= sTop),
                          (df['high'] >= sBot) & (df['high'] <= sTop) & (df['close'] < sBot),
                          pd.Series(False, index=df.index))
    
    s3 = retest_condition(sBreak,
                          (df['high'] >= sBot) & (df['high'] <= sTop),
                          (df['high'] >= sBot) & (df['high'] <= sTop) & (df['close'] < sBot),
                          pd.Series(False, index=df.index),
                          pd.Series(False, index=df.index))
    
    s4 = retest_condition(sBreak,
                          (df['high'] >= sBot) & (df['high'] <= sTop) & (df['close'] < sBot),
                          pd.Series(False, index=df.index),
                          pd.Series(False, index=df.index),
                          pd.Series(False, index=df.index))
    
    # Resistance retest conditions
    r1 = retest_condition(rBreak,
                          (df['low'] <= rBot) & (df['close'] >= rTop),
                          pd.Series(False, index=df.index),
                          pd.Series(False, index=df.index),
                          pd.Series(False, index=df.index))
    
    r2 = retest_condition(rBreak,
                          (df['low'] <= rBot) & (df['close'] <= rTop) & (df['close'] >= rBot),
                          pd.Series(False, index=df.index),
                          pd.Series(False, index=df.index),
                          pd.Series(False, index=df.index))
    
    r3 = retest_condition(rBreak,
                          (df['low'] <= rTop) & (df['low'] >= rBot),
                          pd.Series(False, index=df.index),
                          pd.Series(False, index=df.index),
                          pd.Series(False, index=df.index))
    
    r4 = retest_condition(rBreak,
                          (df['low'] <= rTop) & (df['low'] >= rBot) & (df['close'] > rTop),
                          pd.Series(False, index=df.index),
                          pd.Series(False, index=df.index),
                          pd.Series(False, index=df.index))
    
    # Combine retest conditions
    sRetActive = s1 | s2 | s3 | s4
    rRetActive = r1 | r2 | r3 | r4
    
    # Valuewhen equivalent
    def valuewhen(condition, value, instance=0):
        result = pd.Series(np.nan, index=value.index)
        matches = []
        for i in range(len(condition)):
            if condition.iloc[i]:
                matches.append(value.iloc[i])
            if len(matches) > instance:
                result.iloc[i] = matches[-(instance + 1)]
            else:
                result.iloc[i] = np.nan
        return result
    
    # Calculate retest valid
    sRetEvent = sRetActive & ~sRetActive.shift(1).fillna(False)
    rRetEvent = rRetActive & ~rRetActive.shift(1).fillna(False)
    
    sRetValue = valuewhen(sRetEvent, sTop, 0)
    rRetValue = valuewhen(rRetEvent, rBot, 0)
    
    # Retest validation
    sRetValid = pd.Series(False, index=df.index)
    rRetValid = pd.Series(False, index=df.index)
    
    sRetOccurred = False
    rRetOccurred = False
    
    for i in range(len(df)):
        sBarsSince = barssince(sRetEvent).iloc[i]
        rBarsSince = barssince(rRetEvent).iloc[i]
        
        if pd.notna(sBarsSince) and sBarsSince > 0 and sBarsSince <= input_retValid:
            retConditions = repaint(
                df['close'] >= sRetValue,
                df['high'] >= sRetValue,
                (df['close'] >= sRetValue) & (df['close'] == df['close'])  # barstate.isconfirmed
            )
            if retConditions.iloc[i] and not sRetOccurred:
                sRetValid.iloc[i] = True
                sRetOccurred = True
        
        if pd.notna(rBarsSince) and rBarsSince > 0 and rBarsSince <= input_retValid:
            retConditions = repaint(
                df['close'] <= rRetValue,
                df['low'] <= rRetValue,
                (df['close'] <= rRetValue) & (df['close'] == df['close'])  # barstate.isconfirmed
            )
            if retConditions.iloc[i] and not rRetOccurred:
                rRetValid.iloc[i] = True
                rRetOccurred = True
        
        if pd.notna(sBarsSince) and sBarsSince > input_retValid:
            sRetOccurred = False
        if pd.notna(rBarsSince) and rBarsSince > input_retValid:
            rRetOccurred = False
    
    # Generate entries based on breakout and retest
    for i in range(len(df)):
        # Skip if ATR is NaN (not enough data)
        if pd.isna(atr.iloc[i]):
            continue
        
        # Long entry: support breakout with valid retest
        if sRetValid.iloc[i] and (tradeDirection in ["Long", "Both"]):
            trade_num += 1
            entry_price = df['close'].iloc[i]
            ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
        
        # Short entry: resistance breakout with valid retest
        if rRetValid.iloc[i] and (tradeDirection in ["Short", "Both"]):
            trade_num += 1
            entry_price = df['close'].iloc[i]
            ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
    
    return entries