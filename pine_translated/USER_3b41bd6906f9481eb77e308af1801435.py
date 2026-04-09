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
    
    # Strategy parameters (from Pine Script)
    atrLength = 14
    atrMultiplier = 1.5
    takeProfitRatio = 1.5
    
    # VIDYA inputs
    lengthVidya = 14
    lengthCmo = 14
    alpha = 0.2
    
    # CMO inputs
    cmoLength = 14
    cmoBuyLevel = -50
    cmoSellLevel = 50
    
    # WAE inputs
    waeFastLength = 20
    waeSlowLength = 40
    waeSignalSmoothing = 21
    waeMultiplier = 150
    waeSensitivity = 0.0004
    
    # Break & Retest inputs
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    
    # Trading hours
    startHour = 7
    endHour = 18
    
    # Trade direction
    tradeDirection = "Both"
    
    # Inputs for breakout/retest
    input_breakout = True
    input_retest = True
    
    # Helper function: Wilder RSI
    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Helper function: Wilder ATR
    def wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr
    
    # Helper function: CMO (Chande Momentum Oscillator)
    def calculate_cmo(src, length):
        diff = src.diff()
        up = diff.where(diff > 0, 0.0)
        down = (-diff).where(diff < 0, 0.0)
        up_sum = up.rolling(window=length).sum()
        down_sum = down.rolling(window=length).sum()
        cmo = 100 * (up_sum - down_sum) / (up_sum + down_sum)
        return cmo
    
    # Helper function: VIDYA
    def calculate_vidya(src, length_cmo, alpha):
        cmo_val = calculate_cmo(src, length_cmo)
        vidya = pd.Series(index=src.index, dtype=float)
        vidya.iloc[0] = src.iloc[0]
        for i in range(1, len(src)):
            if pd.isna(cmo_val.iloc[i]):
                vidya.iloc[i] = src.iloc[i]
            else:
                prev_vidya = vidya.iloc[i-1] if not pd.isna(vidya.iloc[i-1]) else src.iloc[i]
                vidya.iloc[i] = prev_vidya + alpha * (cmo_val.iloc[i] / 100) * (src.iloc[i] - prev_vidya)
        return vidya
    
    # Helper function: pivotlow
    def pivot_low(low, bb):
        pivots = pd.Series(index=low.index, dtype=float)
        for i in range(bb, len(low) - bb):
            is_low = True
            for j in range(1, bb + 1):
                if low.iloc[i] >= low.iloc[i - j] or low.iloc[i] >= low.iloc[i + j]:
                    is_low = False
                    break
            if is_low:
                pivots.iloc[i] = low.iloc[i]
            else:
                pivots.iloc[i] = np.nan
        return pivots
    
    # Helper function: pivothigh
    def pivot_high(high, bb):
        pivots = pd.Series(index=high.index, dtype=float)
        for i in range(bb, len(high) - bb):
            is_high = True
            for j in range(1, bb + 1):
                if high.iloc[i] <= high.iloc[i - j] or high.iloc[i] <= high.iloc[i + j]:
                    is_high = False
                    break
            if is_high:
                pivots.iloc[i] = high.iloc[i]
            else:
                pivots.iloc[i] = np.nan
        return pivots
    
    # Helper function: extract hour from timestamp
    def get_hour(ts):
        return datetime.fromtimestamp(ts, tz=timezone.utc).hour
    
    # Calculate indicators
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    time = df['time']
    
    # Calculate CMO
    cmo = calculate_cmo(close, cmoLength)
    
    # Calculate VIDYA
    vidyaValue = calculate_vidya(close, lengthCmo, alpha)
    
    # Calculate ATR
    atr = wilder_atr(high, low, close, atrLength)
    
    # Calculate WAE components
    waeMacd = close.ewm(span=waeFastLength, adjust=False).mean() - close.ewm(span=waeSlowLength, adjust=False).mean()
    waeSignal = waeMacd.ewm(span=waeSignalSmoothing, adjust=False).mean()
    waeAtr = wilder_atr(high, low, close, waeSlowLength) * waeMultiplier
    waeDeadZone = waeSensitivity * volume
    waeExplosionLine = (waeAtr - waeDeadZone).clip(lower=0)
    
    # Calculate pivot points
    bb = input_lookback
    pl = pivot_low(low, bb)
    ph = pivot_high(high, bb)
    
    # Fixnan for pivots
    pl = pl.ffill()
    ph = ph.ffill()
    
    # Box height calculation
    s_yLoc = pd.Series(index=pl.index, dtype=float)
    r_yLoc = pd.Series(index=ph.index, dtype=float)
    
    for i in range(len(df)):
        idx_s = i - bb
        idx_ph = i - bb
        if idx_s >= 0 and idx_s + 1 < len(df):
            s_yLoc.iloc[i] = low.iloc[idx_s + 1] if low.iloc[idx_s + 1] > low.iloc[idx_s - 1] else low.iloc[idx_s - 1]
        else:
            s_yLoc.iloc[i] = np.nan
        if idx_ph >= 0 and idx_ph + 1 < len(df):
            r_yLoc.iloc[i] = high.iloc[idx_ph + 1] if high.iloc[idx_ph + 1] > high.iloc[idx_ph - 1] else high.iloc[idx_ph - 1]
        else:
            r_yLoc.iloc[i] = np.nan
    
    # Initialize breakout tracking
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)
    
    # Calculate box levels (need historical values)
    sBox_top = pd.Series(np.nan, index=df.index)
    sBox_bot = pd.Series(np.nan, index=df.index)
    rBox_top = pd.Series(np.nan, index=df.index)
    rBox_bot = pd.Series(np.nan, index=df.index)
    
    # Track box start and end indices
    box_start_idx = {}
    
    # Function to draw/manage boxes
    def get_box_levels(idx, bb, pl, ph, low, high, s_yLoc, r_yLoc, input_plColor, input_phColor):
        if idx < bb or idx >= len(df) - bb:
            return None, None, None, None
        
        pivot_idx_pl = idx - bb
        pivot_idx_ph = idx - bb
        
        if pivot_idx_pl < 0 or pivot_idx_pl >= len(pl):
            return None, None, None, None
        
        pl_val = pl.iloc[pivot_idx_pl] if not pd.isna(pl.iloc[pivot_idx_pl]) else np.nan
        ph_val = ph.iloc[pivot_idx_ph] if not pd.isna(ph.iloc[pivot_idx_ph]) else np.nan
        
        if pd.isna(pl_val) and pd.isna(ph_val):
            return None, None, None, None
        
        if not pd.isna(pl_val):
            s_top = pl_val
            s_bot = s_yLoc.iloc[idx] if not pd.isna(s_yLoc.iloc[idx]) else pl_val
            return ('support', pivot_idx_pl, s_top, s_bot)
        elif not pd.isna(ph_val):
            r_top = r_yLoc.iloc[idx] if not pd.isna(r_yLoc.iloc[idx]) else ph_val
            r_bot = ph_val
            return ('resistance', pivot_idx_ph, r_top, r_bot)
        
        return None, None, None, None
    
    # Process each bar to detect breakouts
    breakout_up = pd.Series(False, index=df.index)
    breakout_down = pd.Series(False, index=df.index)
    
    # Track active boxes and breakouts
    active_boxes = {}  # idx -> {'type': 'support'/'resistance', 'top': float, 'bot': float, 'start_idx': int}
    
    for i in range(bb + 1, len(df)):
        # Check for new pivot point
        if i - bb >= 0 and i - bb < len(pl):
            pl_val = pl.iloc[i - bb]
            if not pd.isna(pl_val) and i - bb not in [k - bb for k in active_boxes.keys() if k - bb >= 0]:
                # New support pivot
                s_top = pl_val
                s_bot = s_yLoc.iloc[i] if not pd.isna(s_yLoc.iloc[i]) else pl_val
                active_boxes[i - bb] = {'type': 'support', 'top': s_top, 'bot': s_bot, 'start_idx': i - bb}
        
        if i - bb >= 0 and i - bb < len(ph):
            ph_val = ph.iloc[i - bb]
            if not pd.isna(ph_val) and i - bb not in [k - bb for k in active_boxes.keys() if k - bb >= 0]:
                # New resistance pivot
                r_top = r_yLoc.iloc[i] if not pd.isna(r_yLoc.iloc[i]) else ph_val
                r_bot = ph_val
                active_boxes[i - bb] = {'type': 'resistance', 'top': r_top, 'bot': r_bot, 'start_idx': i - bb}
        
        # Check for breakout conditions
        for box_idx, box_data in list(active_boxes.items()):
            box_end = box_idx + bb
            
            if box_data['type'] == 'support':
                sBot = box_data['bot']
                # Crossunder detection
                if i > 0 and close.iloc[i] < sBot and close.iloc[i-1] >= sBot:
                    sBreak.iloc[i] = True
                    breakout_down.iloc[i] = True
                    del active_boxes[box_idx]
            
            elif box_data['type'] == 'resistance':
                rTop = box_data['top']
                # Crossover detection
                if i > 0 and close.iloc[i] > rTop and close.iloc[i-1] <= rTop:
                    rBreak.iloc[i] = True
                    breakout_up.iloc[i] = True
                    del active_boxes[box_idx]
    
    # Calculate retest conditions
    # s1: high >= sTop and close <= sBot
    # s2: high >= sTop and close >= sBot and close <= sTop
    # s3: high >= sBot and high <= sTop
    # s4: high >= sBot and high <= sTop and close < sBot
    
    # r1: low <= rBot and close >= rTop
    # r2: low <= rBot and close <= rTop and close >= rBot
    # r3: low <= rTop and low >= rBot
    # r4: low <= rTop and low >= rBot and close > rTop
    
    s1 = pd.Series(False, index=df.index)
    s2 = pd.Series(False, index=df.index)
    s3 = pd.Series(False, index=df.index)
    s4 = pd.Series(False, index=df.index)
    r1 = pd.Series(False, index=df.index)
    r2 = pd.Series(False, index=df.index)
    r3 = pd.Series(False, index=df.index)
    r4 = pd.Series(False, index=df.index)
    
    # Store retest box reference per breakout
    sBreak_boxes = {}  # bar_idx -> {'top': float, 'bot': float}
    rBreak_boxes = {}  # bar_idx -> {'top': float, 'bot': float}
    
    # Track active boxes with breakout info
    active_boxes_with_break = {}
    
    for i in range(bb + 1, len(df)):
        if breakout_down.iloc[i]:
            # Find the support box that was broken
            for box_idx, box_data in list(active_boxes.items()):
                if box_data['type'] == 'support':
                    sBreak_boxes[i] = {'top': box_data['top'], 'bot': box_data['bot']}
                    active_boxes_with_break[i] = box_data
                    break
        
        if breakout_up.iloc[i]:
            # Find the resistance box that was broken
            for box_idx, box_data in list(active_boxes.items()):
                if box_data['type'] == 'resistance':
                    rBreak_boxes[i] = {'top': box_data['top'], 'bot': box_data['bot']}
                    active_boxes_with_break[i] = box_data
                    break
    
    # Re-calculate with proper box tracking
    # Reset tracking
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)
    breakout_up = pd.Series(False, index=df.index)
    breakout_down = pd.Series(False, index=df.index)
    
    boxes = []  # List of {'start_idx': int, 'type': str, 'top': float, 'bot': float}
    sBreak_info = {}  # bar_idx -> {'top': float, 'bot': float}
    rBreak_info = {}  # bar_idx -> {'top': float, 'bot': float}
    
    # First pass: identify all boxes from pivot points
    for i in range(bb, len(df) - bb):
        pivot_idx = i - bb
        if pivot_idx < 0 or pivot_idx >= len(pl):
            continue
        
        pl_val = pl.iloc[pivot_idx]
        ph_val = ph.iloc[pivot_idx]
        
        if not pd.isna(pl_val):
            # Create support box
            s_top = pl_val
            s_bot = s_yLoc.iloc[i] if not pd.isna(s_yLoc.iloc[i]) else pl_val
            # Check if box already exists at this position
            existing = [b for b in boxes if b['start_idx'] == pivot_idx and b['type'] == 'support']
            if not existing:
                boxes.append({'start_idx': pivot_idx, 'type': 'support', 'top': s_top, 'bot': s_bot})
        
        if not pd.isna(ph_val):
            # Create resistance box
            r_top = r_yLoc.iloc[i] if not pd.isna(r_yLoc.iloc[i]) else ph_val
            r_bot = ph_val
            existing = [b for b in boxes if b['start_idx'] == pivot_idx and b['type'] == 'resistance']
            if not existing:
                boxes.append({'start_idx': pivot_idx, 'type': 'resistance', 'top': r_top, 'bot': r_bot})
    
    # Second pass: detect breakouts and calculate retest conditions
    for i in range(bb, len(df)):
        for box in boxes:
            start_idx = box['start_idx']
            end_idx = start_idx + bb
            
            if i != end_idx:
                continue
            
            # Box is at this bar
            if box['type'] == 'support':
                sBot = box['bot']
                sTop = box['top']
                
                # Check for crossunder (support breakout)
                if i > 0 and close.iloc[i] < sBot and close.iloc[i-1] >= sBot:
                    sBreak.iloc[i] = True
                    breakout_down.iloc[i] = True
                    sBreak_info[i] = {'top': sTop, 'bot': sBot}
                    # Remove this box as it's been broken
                    boxes = [b for b in boxes if b != box]
                    break
            
            elif box['type'] == 'resistance':
                rTop = box['top']
                rBot = box['bot']
                
                # Check for crossover (resistance breakout)
                if i > 0 and close.iloc[i] > rTop and close.iloc[i-1] <= rTop:
                    rBreak.iloc[i] = True
                    breakout_up.iloc[i] = True
                    rBreak_info[i] = {'top': rTop, 'bot': rBot}
                    # Remove this box as it's been broken
                    boxes = [b for b in boxes if b != box]
                    break
    
    # Calculate bars since breakout
    bars_since_sBreak = pd.Series(0, index=df.index)
    bars_since_rBreak = pd.Series(0, index=df.index)
    
    last_sBreak_idx = -1
    last_rBreak_idx = -1
    
    for i in range(len(df)):
        if sBreak.iloc[i]:
            last_sBreak_idx = i
            bars_since_sBreak.iloc[i] = 0
        elif last_sBreak_idx >= 0:
            bars_since_sBreak.iloc[i] = i - last_sBreak_idx
        else:
            bars_since_sBreak.iloc[i] = 999
        
        if rBreak.iloc[i]:
            last_rBreak_idx = i
            bars_since_rBreak.iloc[i] = 0
        elif last_rBreak_idx >= 0:
            bars_since_rBreak.iloc[i] = i - last_rBreak_idx
        else:
            bars_since_rBreak.iloc[i] = 999
    
    # Calculate retest conditions with sBreak_info and rBreak_info
    for i in range(len(df)):
        if i in sBreak_info:
            box = sBreak_info[i]
            sTop = box['top']
            sBot = box['bot']
            
            # Reset conditions after new breakout
            s1.iloc[i] = False
            s2.iloc[i] = False
            s3.iloc[i] = False
            s4.iloc[i] = False
            
            # Continue checking in subsequent bars
            for j in range(i + 1, min(i + input_retValid + 1, len(df))):
                if bars_since_sBreak.iloc[j] > input_retSince:
                    # s1: high >= sTop and close <= sBot
                    if high.iloc[j] >= sTop and close.iloc[j] <= sBot:
                        s1.iloc[j] = True
                    # s2: high >= sTop and close >= sBot and close <= sTop
                    if high.iloc[j] >= sTop and close.iloc[j] >= sBot and close.iloc[j] <= sTop:
                        s2.iloc[j] = True
                    # s3: high >= sBot and high <= sTop
                    if high.iloc[j] >= sBot and high.iloc[j] <= sTop:
                        s3.iloc[j] = True
                    # s4: high >= sBot and high <= sTop and close < sBot
                    if high.iloc[j] >= sBot and high.iloc[j] <= sTop and close.iloc[j] < sBot:
                        s4.iloc[j] = True
        
        if i in rBreak_info:
            box = rBreak_info[i]
            rTop = box['top']
            rBot = box['bot']
            
            # r1: low <= rBot and close >= rTop
            if low.iloc[i] <= rBot and close.iloc[i] >= rTop:
                r1.iloc[i] = True
            # r2: low <= rBot and close <= rTop and close >= rBot
            if low.iloc[i] <= rBot and close.iloc[i] <= rTop and close.iloc[i] >= rBot:
                r2.iloc[i] = True
            # r3: low <= rTop and low >= rBot
            if low.iloc[i] <= rTop and low.iloc[i] >= rBot:
                r3.iloc[i] = True
            # r4: low <= rTop and low >= rBot and close > rTop
            if low.iloc[i] <= rTop and low.iloc[i] >= rBot and close.iloc[i] > rTop:
                r4.iloc[i] = True
            
            # Continue checking in subsequent bars
            for j in range(i + 1, min(i + input_retValid + 1, len(df))):
                if bars_since_rBreak.iloc[j] > input_retSince:
                    if r1.iloc[j - 1]:
                        pass
                    # r1: low <= rBot and close >= rTop
                    if low.iloc[j] <= rBot and close.iloc[j] >= rTop:
                        r1.iloc[j] = True
                    # r2: low <= rBot and close <= rTop and close >= rBot
                    if low.iloc[j] <= rBot and close.iloc[j] <= rTop and close.iloc[j] >= rBot:
                        r2.iloc[j] = True
                    # r3: low <= rTop and low >= rBot
                    if low.iloc[j] <= rTop and low.iloc[j] >= rBot:
                        r3.iloc[j] = True
                    # r4: low <= rTop and low >= rBot and close > rTop
                    if low.iloc[j] <= rTop and low.iloc[j] >= rBot and close.iloc[j] > rTop:
                        r4.iloc[j] = True
    
    # Combine retest conditions
    long_retest = s1 | s2 | s3 | s4
    short_retest = r1 | r2 | r3 | r4
    
    # Trading hours check
    trading_hours = pd.Series(index=df.index, dtype=bool)
    for i in range(len(df)):
        hour = get_hour(time.iloc[i])
        trading_hours.iloc[i] = startHour <= hour < endHour
    
    # Check for valid lookahead bar (need bb+1 bars after entry for pivot detection)
    valid_bar = df.index >= bb
    
    # Build entry conditions based on trade direction
    long_entry = pd.Series(False, index=df.index)
    short_entry = pd.Series(False, index=df.index)
    
    for i in range(len(df)):
        if not valid_bar.iloc[i]:
            continue
        if not trading_hours.iloc[i]:
            continue
        
        # Long entry: need sBreak and then retest condition met
        if sBreak.iloc[i]:
            # Check subsequent bars for retest
            for j in range(i + 1, min(i + input_retValid + 2, len(df))):
                if bars_since_sBreak.iloc[j] > input_retSince:
                    if long_retest.iloc[j]:
                        long_entry.iloc[j] = True
                        # Reset sBreak tracking to prevent duplicate entries
                        # Only trigger once per breakout
                        break
                elif bars_since_sBreak.iloc[j] > input_retValid + input_retSince:
                    break
        
        # Short entry: need rBreak and then retest condition met
        if rBreak.iloc[i]:
            # Check subsequent bars for retest
            for j in range(i + 1, min(i + input_retValid + 2, len(df))):
                if bars_since_rBreak.iloc[j] > input_retSince:
                    if short_retest.iloc[j]:
                        short_entry.iloc[j] = True
                        break
                elif bars_since_rBreak.iloc[j] > input_retValid + input_retSince:
                    break
    
    # Apply trade direction filter
    if tradeDirection == "Long":
        short_entry = pd.Series(False, index=df.index)
    elif tradeDirection == "Short":
        long_entry = pd.Series(False, index=df.index)
    
    # Generate entry list
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_entry.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = int(time.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
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
        
        if short_entry.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = int(time.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            
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
    
    return entries