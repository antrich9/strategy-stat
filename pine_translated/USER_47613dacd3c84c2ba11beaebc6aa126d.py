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
    # Strategy parameters
    lookback = 50
    mit = "full fill high/low"
    gap = 0.0
    atrLength = 14
    
    # Bar resolution in ms (default to 1min = 60000ms)
    bar = 60000
    
    entries = []
    trade_num = 0
    
    # Track active FVGs: list of [price, timestamp]
    bullfvghigh = []
    bullfvglow = []
    bullfvghight = []
    bullfvglowt = []
    
    bearfvghigh = []
    bearfvglow = []
    bearfvghight = []
    bearfvglowt = []
    
    for i in range(len(df)):
        ts = int(df['time'].iloc[i])
        curr_open = df['open'].iloc[i]
        curr_high = df['high'].iloc[i]
        curr_low = df['low'].iloc[i]
        curr_close = df['close'].iloc[i]
        
        # Previous bars
        prev_high = df['high'].iloc[i-1] if i >= 1 else curr_high
        prev_low = df['low'].iloc[i-1] if i >= 1 else curr_low
        prev2_high = df['high'].iloc[i-2] if i >= 2 else prev_high
        prev2_low = df['low'].iloc[i-2] if i >= 2 else prev_low
        prev2_time = df['time'].iloc[i-2] if i >= 2 else ts
        
        bullSignal = False
        bearSignal = False
        
        if i >= 2:
            # Detect Bull FVG: high[2] < low AND gap condition
            gap_pct = (abs(prev2_high - curr_low) / prev2_high) * 100 if prev2_high != 0 else 0
            if prev2_high < curr_low and gap_pct > gap:
                bullfvghigh.append(prev2_high)
                bullfvglow.append(curr_low)
                bullfvghight.append(prev2_time)
                bullfvglowt.append(ts)
            
            # Detect Bear FVG: low[2] > high AND gap condition
            gap_pct_bear = (abs(curr_high - prev2_low) / curr_high) * 100 if curr_high != 0 else 0
            if prev2_low > curr_high and gap_pct_bear > gap:
                bearfvghigh.append(curr_high)
                bearfvglow.append(prev2_low)
                bearfvghight.append(ts)
                bearfvglowt.append(prev2_time)
        
        # Mitigation check - remove mitigated FVGs
        cutoff_time = ts - lookback * (bar + 2)
        
        # Bull FVG mitigation
        if mit == "full fill high/low":
            remaining_bull = []
            for x in range(len(bullfvghigh)):
                # Condition: not(low <= bullfvghigh[x]) AND within lookback
                if not (curr_low <= bullfvghigh[x]) and bullfvghight[x] >= cutoff_time:
                    remaining_bull.append(x)
            bullfvghigh = [bullfvghigh[x] for x in remaining_bull]
            bullfvglow = [bullfvglow[x] for x in remaining_bull]
            bullfvghight = [bullfvghight[x] for x in remaining_bull]
            bullfvglowt = [bullfvglowt[x] for x in remaining_bull]
        
        # Bear FVG mitigation
        if mit == "full fill high/low":
            remaining_bear = []
            for x in range(len(bearfvghigh)):
                # Condition: not(high >= bearfvglow[x]) AND within lookback
                if not (curr_high >= bearfvglow[x]) and bearfvglowt[x] >= cutoff_time:
                    remaining_bear.append(x)
            bearfvghigh = [bearfvghigh[x] for x in remaining_bear]
            bearfvglow = [bearfvglow[x] for x in remaining_bear]
            bearfvghight = [bearfvghight[x] for x in remaining_bear]
            bearfvglowt = [bearfvglowt[x] for x in remaining_bear]
        
        # BPR detection
        if len(bullfvghigh) > 0 and len(bearfvghigh) > 0:
            for i_bear in range(len(bearfvghigh)):
                for j_bull in range(len(bullfvghigh)):
                    # Bull Signal: bearfvghigh[i] < bullfvglow[j] AND bearfvglow[i] > bullfvghigh[j]
                    if bearfvghigh[i_bear] < bullfvglow[j_bull] and bearfvglow[i_bear] > bullfvghigh[j_bull]:
                        bullSignal = True
                        break
                    # Bear Signal: bearfvghigh[i] > bullfvglow[j] AND bearfvglow[i] < bullfvghigh[j]
                    elif bearfvghigh[i_bear] > bullfvglow[j_bull] and bearfvglow[i_bear] < bullfvghigh[j_bull]:
                        bearSignal = True
                        break
                if bullSignal or bearSignal:
                    break
        
        # Generate entries
        if bullSignal:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
        
        if bearSignal:
            trade_num += 1
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
    
    return entries