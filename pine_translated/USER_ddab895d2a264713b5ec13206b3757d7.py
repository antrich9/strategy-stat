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
    
    entries = []
    trade_num = 1
    
    n = len(df)
    if n < 5:
        return entries
    
    time_arr = df['time'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    close_arr = df['close'].values
    
    swing_length = 5
    wick_threshold = 0.3
    atr_length = 14
    
    # Calculate swing high/low using pivot logic
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    
    for i in range(swing_length, n - swing_length):
        is_high = True
        is_low = True
        for j in range(1, swing_length + 1):
            if high_arr[i] <= high_arr[i - j] or high_arr[i] <= high_arr[i + j]:
                is_high = False
            if low_arr[i] >= low_arr[i - j] or low_arr[i] >= low_arr[i + j]:
                is_low = False
        if is_high:
            swing_high[i] = high_arr[i]
        if is_low:
            swing_low[i] = low_arr[i]
    
    # Calculate ATR using Wilder's method
    tr = np.full(n, np.nan)
    tr[1:] = np.maximum(high_arr[1:] - low_arr[1:], 
                        np.maximum(np.abs(high_arr[1:] - close_arr[:-1]),
                                   np.abs(low_arr[1:] - close_arr[:-1])))
    
    atr = np.full(n, np.nan)
    atr[atr_length] = np.mean(tr[1:atr_length + 1])
    for i in range(atr_length + 1, n):
        atr[i] = (atr[i - 1] * (atr_length - 1) + tr[i]) / atr_length
    
    # Detect FVGs
    bullish_fvg = np.full(n, False)
    bearish_fvg = np.full(n, False)
    
    for i in range(2, n):
        if low_arr[i - 1] > high_arr[i - 2]:
            bullish_fvg[i] = True
        if high_arr[i - 1] < low_arr[i - 2]:
            bearish_fvg[i] = True
    
    # Detect bar closed (change in time)
    bar_closed = np.full(n, False)
    for i in range(1, n):
        bar_closed[i] = (time_arr[i] // 86400000) != (time_arr[i - 1] // 86400000)
    
    # PDA class simulation
    class PDA:
        def __init__(self, pda_type, top, bottom, mid_level, bar_created, is_premium, rejection_state):
            self.pda_type = pda_type
            self.top = top
            self.bottom = bottom
            self.mid_level = mid_level
            self.bar_created = bar_created
            self.is_premium = is_premium
            self.rejection_state = rejection_state
            self.first_candle_high = np.nan
            self.first_candle_low = np.nan
            self.rejection_level = np.nan
            self.rejection_bar = -1
    
    all_pdas = []
    
    def is_bullish_rejection(h, l, o, c):
        candle_range = h - l
        if candle_range == 0:
            return False
        lower_wick = min(c, o) - l
        return (lower_wick / candle_range) >= wick_threshold
    
    def is_bearish_rejection(h, l, o, c):
        candle_range = h - l
        if candle_range == 0:
            return False
        upper_wick = h - max(c, o)
        return (upper_wick / candle_range) >= wick_threshold
    
    def price_touching_pda(pda, price_high, price_low):
        return price_high >= pda.bottom and price_low <= pda.top
    
    # Process bars
    for i in range(n):
        current_high = high_arr[i]
        current_low = low_arr[i]
        current_open = open_arr[i]
        current_close = close_arr[i]
        
        # Detect new PDAs on bar close
        if i > 0 and bar_closed[i]:
            # Bullish FVG using previous bars
            if i >= 2:
                if low_arr[i - 1] > high_arr[i - 2]:
                    pda = PDA("BULL_FVG", current_close, current_close, (low_arr[i - 1] + high_arr[i - 2]) / 2, 
                              i, False, "PENDING")
                    pda.top = low_arr[i - 1]
                    pda.bottom = high_arr[i - 2]
                    all_pdas.append(pda)
                
                if high_arr[i - 1] < low_arr[i - 2]:
                    pda = PDA("BEAR_FVG", low_arr[i - 2], high_arr[i - 1], (low_arr[i - 2] + high_arr[i - 1]) / 2, 
                              i, True, "PENDING")
                    pda.top = high_arr[i - 1]
                    pda.bottom = low_arr[i - 2]
                    all_pdas.append(pda)
            
            # Swing High
            if not np.isnan(swing_high[i]):
                pda = PDA("SWING_HIGH", swing_high[i] * 1.0005, swing_high[i] * 0.9995, 
                          swing_high[i], i, True, "PENDING")
                pda.top = swing_high[i] * 1.0005
                pda.bottom = swing_high[i] * 0.9995
                all_pdas.append(pda)
            
            # Swing Low
            if not np.isnan(swing_low[i]):
                pda = PDA("SWING_LOW", swing_low[i] * 1.0005, swing_low[i] * 0.9995, 
                          swing_low[i], i, False, "PENDING")
                pda.top = swing_low[i] * 1.0005
                pda.bottom = swing_low[i] * 0.9995
                all_pdas.append(pda)
        
        # Process existing PDAs
        pdas_to_remove = []
        
        for idx, pda in enumerate(all_pdas):
            # Remove old PDAs
            if i - pda.bar_created > 100:
                pdas_to_remove.append(idx)
                continue
            
            if pda.rejection_state in ["DISRESPECTED", "PROCESSED"]:
                continue
            
            # Check if price is touching PDA
            is_touching = price_touching_pda(pda, current_high, current_low)
            
            if pda.rejection_state == "PENDING" and is_touching:
                pda.rejection_state = "CANDLE_1"
                pda.first_candle_high = current_high
                pda.first_candle_low = current_low
                
                # Check rejection on current bar
                if pda.is_premium and is_bearish_rejection(current_high, current_low, current_open, current_close):
                    pda.rejection_level = current_high
                    pda.rejection_bar = i
                    pda.rejection_state = "RESPECTING"
                    # Generate SHORT entry
                    entry_price = current_close
                    entry_ts = int(time_arr[i])
                    entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
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
                    pda.rejection_state = "PROCESSED"
                elif not pda.is_premium and is_bullish_rejection(current_high, current_low, current_open, current_close):
                    pda.rejection_level = current_low
                    pda.rejection_bar = i
                    pda.rejection_state = "RESPECTING"
                    # Generate LONG entry
                    entry_price = current_close
                    entry_ts = int(time_arr[i])
                    entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
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
                    pda.rejection_state = "PROCESSED"
            
            elif pda.rejection_state == "CANDLE_1" and bar_closed[i]:
                pda.rejection_state = "CANDLE_2"
                
                if pda.is_premium:
                    if current_high > pda.first_candle_high and is_bearish_rejection(current_high, current_low, current_open, current_close):
                        pda.rejection_level = current_high
                        pda.rejection_bar = i
                        pda.rejection_state = "RESPECTING"
                        # Generate SHORT entry
                        entry_price = current_close
                        entry_ts = int(time_arr[i])
                        entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
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
                        pda.rejection_state = "PROCESSED"
                else:
                    if current_low < pda.first_candle_low and is_bullish_rejection(current_high, current_low, current_open, current_close):
                        pda.rejection_level = current_low
                        pda.rejection_bar = i
                        pda.rejection_state = "RESPECTING"
                        # Generate LONG entry
                        entry_price = current_close
                        entry_ts = int(time_arr[i])
                        entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
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
                        pda.rejection_state = "PROCESSED"
            
            elif pda.rejection_state == "CANDLE_2" and bar_closed[i]:
                if np.isnan(pda.rejection_level):
                    pda.rejection_state = "DISRESPECTED"
        
        # Remove old PDAs in reverse order
        for idx in sorted(pdas_to_remove, reverse=True):
            if idx < len(all_pdas):
                all_pdas.pop(idx)
    
    return entries