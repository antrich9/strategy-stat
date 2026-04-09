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
    
    # Parameters (matching Pine Script defaults)
    atrLength = 14
    atrMultiplier = 1.5
    takeProfitRatio = 1.5
    tradeDirection = 'Both'
    lengthVidya = 14
    lengthCmo = 14
    alpha = 0.2
    cmoLength = 14
    cmoBuyLevel = -50
    cmoSellLevel = 50
    shortTermVolLength = 5
    longTermVolLength = 20
    volOscThreshold = 0
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    input_breakout = True
    input_retest = True
    input_repType = 'On'
    
    startHour = 7
    endHour = 18
    
    entries = []
    trade_num = 1
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    times = df['time'].values
    
    n = len(df)
    bb = input_lookback
    
    # CMO (Wilder RSI-like)
    diff = np.diff(close, prepend=close[0])
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1.0/cmoLength, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1.0/cmoLength, adjust=False).mean().values
    avg_loss = np.where(avg_loss == 0, np.nan, avg_loss)
    rs = avg_gain / avg_loss
    cmo = 100 * (1 - 1 / (1 + rs))
    cmo = np.nan_to_num(cmo, nan=0.0)
    
    # VIDYA
    vidya = np.zeros(n)
    for i in range(n):
        if i == 0:
            vidya[i] = close[i]
        else:
            cmo_val = cmo[i] / 100
            vidya[i] = vidya[i-1] + alpha * cmo_val * (close[i] - vidya[i-1])
    
    # Volume Oscillator
    vol_osc = pd.Series(volume).rolling(shortTermVolLength).mean().fillna(0).values - \
              pd.Series(volume).rolling(longTermVolLength).mean().fillna(0).values
    
    # ATR (Wilder)
    hl = high - low
    hc = np.abs(high - np.roll(close, 1))
    lc = np.abs(low - np.roll(close, 1))
    hc[0] = 0
    lc[0] = 0
    tr = np.maximum(hl, np.maximum(hc, lc))
    atr = pd.Series(tr).ewm(alpha=1.0/atrLength, adjust=False).mean().values
    
    # Pivot calculations
    pl = np.full(n, np.nan)
    ph = np.full(n, np.nan)
    
    for i in range(bb, n - bb):
        left_len = bb
        right_len = bb
        local_low_idx = np.argmin(low[i - left_len:i + right_len + 1]) + (i - left_len)
        local_high_idx = np.argmax(high[i - left_len:i + right_len + 1]) + (i - left_len)
        if local_low_idx == i:
            pl[i] = low[i]
        if local_high_idx == i:
            ph[i] = high[i]
    
    pl = pd.Series(pl).ffill().values
    ph = pd.Series(ph).ffill().values
    
    # Box boundaries
    s_yLoc = np.zeros(n)
    r_yLoc = np.zeros(n)
    for i in range(bb, n):
        idx1 = i - bb - 1
        idx2 = i - bb + 1
        if idx1 >= 0 and idx2 < n:
            s_yLoc[i] = low[idx1] if low[idx1] > low[idx2] else low[idx2]
            r_yLoc[i] = high[idx1] if high[idx1] > high[idx2] else high[idx2]
    
    # Box tracking
    sBox_top = np.full(n, np.nan)
    sBox_bottom = np.full(n, np.nan)
    rBox_top = np.full(n, np.nan)
    rBox_bottom = np.full(n, np.nan)
    
    active_s_box_top = np.nan
    active_s_box_bottom = np.nan
    active_r_box_top = np.nan
    active_r_box_bottom = np.nan
    
    for i in range(bb + 1, n):
        if i > 0 and not np.isnan(pl[i]) and pl[i] != pl[i-1]:
            active_s_box_bottom = pl[i]
            active_s_box_top = s_yLoc[i]
        if i > 0 and not np.isnan(ph[i]) and ph[i] != ph[i-1]:
            active_r_box_top = ph[i]
            active_r_box_bottom = r_yLoc[i]
        
        sBox_top[i] = active_s_box_top
        sBox_bottom[i] = active_s_box_bottom
        rBox_top[i] = active_r_box_top
        rBox_bottom[i] = active_r_box_bottom
    
    # Breakout tracking
    sBreak = np.zeros(n, dtype=bool)
    rBreak = np.zeros(n, dtype=bool)
    
    cu = np.zeros(n, dtype=bool)
    co = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        s_bot = sBox_bottom[i]
        r_top = rBox_top[i]
        
        if not np.isnan(s_bot) and not np.isnan(s_bot):
            if close[i] < s_bot and close[i-1] >= s_bot:
                cu[i] = True
                sBreak[i] = True
        
        if not np.isnan(r_top) and not np.isnan(r_top):
            if close[i] > r_top and close[i-1] <= r_top:
                co[i] = True
                rBreak[i] = True
    
    # Retest event tracking
    retOccurred = np.zeros(n, dtype=bool)
    
    # Trading hours check
    trading_hours = np.zeros(n, dtype=bool)
    for i in range(n):
        dt = datetime.fromtimestamp(times[i], tz=timezone.utc)
        trading_hours[i] = startHour <= dt.hour < endHour
    
    # Generate entries
    for i in range(bb + 1, n):
        # Skip if not trading hours
        if not trading_hours[i]:
            continue
        
        # Skip if volume oscillator below threshold
        if vol_osc[i] < volOscThreshold:
            continue
        
        s_bot = sBox_bottom[i]
        s_top = sBox_top[i]
        r_bot = rBox_bottom[i]
        r_top = rBox_top[i]
        
        # Support retest conditions
        s1 = False
        s2 = False
        s3 = False
        s4 = False
        
        if not np.isnan(s_bot) and not np.isnan(s_top) and sBreak[i]:
            bars_since = i - np.where(sBreak[:i+1])[0][-1] if np.any(sBreak[:i+1]) else 999
            if bars_since > input_retSince:
                if not np.isnan(s_bot):
                    cond1 = high[i] >= s_top and close[i] <= s_bot
                    cond2 = high[i] >= s_top and close[i] >= s_bot and close[i] <= s_top
                    cond3 = high[i] >= s_bot and high[i] <= s_top
                    cond4 = high[i] >= s_bot and high[i] <= s_top and close[i] < s_bot
                    s1 = cond1
                    s2 = cond2
                    s3 = cond3
                    s4 = cond4
        
        # Resistance retest conditions
        r1 = False
        r2 = False
        r3 = False
        r4 = False
        
        if not np.isnan(r_bot) and not np.isnan(r_top) and rBreak[i]:
            bars_since_r = i - np.where(rBreak[:i+1])[0][-1] if np.any(rBreak[:i+1]) else 999
            if bars_since_r > input_retSince:
                if not np.isnan(r_bot):
                    cond1 = low[i] <= r_bot and close[i] >= r_top
                    cond2 = low[i] <= r_bot and close[i] <= r_top and close[i] >= r_bot
                    cond3 = low[i] <= r_top and low[i] >= r_bot
                    cond4 = low[i] <= r_top and low[i] >= r_bot and close[i] > r_top
                    r1 = cond1
                    r2 = cond2
                    r3 = cond3
                    r4 = cond4
        
        # Retest event detection
        retActive = s1 or s2 or s3 or s4 or r1 or r2 or r3 or r4
        retEvent = retActive and (i == 0 or not retActive)
        
        if retEvent:
            retOccurred[i] = True
        
        # Long entry (support retest)
        if input_retest and (s1 or s2 or s3 or s4) and not retOccurred[i]:
            if i > 0:
                bars_since_s = i - np.where(sBreak[:i+1])[0][-1] if np.any(sBreak[:i+1]) else 999
                ret_valid = bars_since_s > 0 and bars_since_s <= input_retValid
                if ret_valid:
                    if tradeDirection in ['Long', 'Both']:
                        entries.append({
                            'trade_num': trade_num,
                            'direction': 'long',
                            'entry_ts': int(times[i]),
                            'entry_time': datetime.fromtimestamp(times[i], tz=timezone.utc).isoformat(),
                            'entry_price_guess': float(close[i]),
                            'exit_ts': 0,
                            'exit_time': '',
                            'exit_price_guess': 0.0,
                            'raw_price_a': float(close[i]),
                            'raw_price_b': float(close[i])
                        })
                        trade_num += 1
                    retOccurred[i] = True
        
        # Short entry (resistance retest)
        if input_retest and (r1 or r2 or r3 or r4) and not retOccurred[i]:
            if i > 0:
                bars_since_r = i - np.where(rBreak[:i+1])[0][-1] if np.any(rBreak[:i+1]) else 999
                ret_valid = bars_since_r > 0 and bars_since_r <= input_retValid
                if ret_valid:
                    if tradeDirection in ['Short', 'Both']:
                        entries.append({
                            'trade_num': trade_num,
                            'direction': 'short',
                            'entry_ts': int(times[i]),
                            'entry_time': datetime.fromtimestamp(times[i], tz=timezone.utc).isoformat(),
                            'entry_price_guess': float(close[i]),
                            'exit_ts': 0,
                            'exit_time': '',
                            'exit_price_guess': 0.0,
                            'raw_price_a': float(close[i]),
                            'raw_price_b': float(close[i])
                        })
                        trade_num += 1
                    retOccurred[i] = True
    
    return entries