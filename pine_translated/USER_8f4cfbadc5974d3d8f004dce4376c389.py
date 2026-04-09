import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    """
    results = []
    
    high = df['high']
    low = df['low']
    close = df['close']
    timestamps = df['time']
    
    # Calculate FVG conditions (using close[1] for previous bar close)
    bull_fvg = (low > high.shift(2)) & (close.shift(1) > high.shift(2))
    bear_fvg = (high < low.shift(2)) & (close.shift(1) < low.shift(2))
    
    # FVG zone boundaries
    bull_top = low.copy()
    bull_bottom = high.shift(2).copy()
    bear_top = low.shift(2).copy()
    bear_bottom = high.copy()
    
    # Initialize tracking arrays
    bull_fvg_active = bull_fvg.copy()
    bear_fvg_active = bear_fvg.copy()
    bull_fvg_filled = pd.Series(False, index=df.index)
    bear_fvg_filled = pd.Series(False, index=df.index)
    bull_fvg_ts = pd.Series(0, index=df.index, dtype=np.int64)
    bear_fvg_ts = pd.Series(0, index=df.index, dtype=np.int64)
    
    lookback = 12
    
    # Find barsSince equivalents and set up FVG tracking
    for i in range(2, len(df)):
        if bull_fvg.iloc[i]:
            bar_count = 0
            for j in range(i - 1, 1, -1):
                if bear_fvg.iloc[j]:
                    bar_count = i - j
                    break
            if bar_count > 0 and bar_count <= lookback:
                bull_fvg_active.iloc[i] = True
                bull_fvg_ts.iloc[i] = timestamps.iloc[i]
            else:
                bull_fvg_active.iloc[i] = False
        
        if bear_fvg.iloc[i]:
            bar_count = 0
            for j in range(i - 1, 1, -1):
                if bull_fvg.iloc[j]:
                    bar_count = i - j
                    break
            if bar_count > 0 and bar_count <= lookback:
                bear_fvg_active.iloc[i] = True
                bear_fvg_ts.iloc[i] = timestamps.iloc[i]
            else:
                bear_fvg_active.iloc[i] = False
    
    # Find entry signals
    bull_entry = pd.Series(False, index=df.index)
    bear_entry = pd.Series(False, index=df.index)
    
    for i in range(2, len(df)):
        if bull_fvg_active.iloc[i]:
            for j in range(i - 1, 1, -1):
                if bull_fvg_active.iloc[j] and not bull_fvg_filled.iloc[j]:
                    bull_fvg_top = bull_top.iloc[j]
                    if low.iloc[i] < bull_fvg_top:
                        bull_entry.iloc[i] = True
                        bull_fvg_filled.iloc[j] = True
                    break
        
        if bear_fvg_active.iloc[i]:
            for j in range(i - 1, 1, -1):
                if bear_fvg_active.iloc[j] and not bear_fvg_filled.iloc[j]:
                    bear_fvg_bottom = bear_bottom.iloc[j]
                    if high.iloc[i] > bear_fvg_bottom:
                        bear_entry.iloc[i] = True
                        bear_fvg_filled.iloc[j] = True
                    break
    
    trade_num = 1
    
    for i in range(len(df)):
        if bull_entry.iloc[i]:
            entry_ts = int(timestamps.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
            for j in range(i - 1, 1, -1):
                if bull_fvg_active.iloc[j]:
                    results.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': entry_ts,
                        'entry_time': entry_time,
                        'entry_price_guess': entry_price,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(bull_top.iloc[j]),
                        'raw_price_b': float(bull_bottom.iloc[j])
                    })
                    trade_num += 1
                    break
        
        if bear_entry.iloc[i]:
            entry_ts = int(timestamps.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
            for j in range(i - 1, 1, -1):
                if bear_fvg_active.iloc[j]:
                    results.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': entry_ts,
                        'entry_time': entry_time,
                        'entry_price_guess': entry_price,
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(bear_top.iloc[j]),
                        'raw_price_b': float(bear_bottom.iloc[j])
                    })
                    trade_num += 1
                    break
    
    results.sort(key=lambda x: x['entry_ts'])
    for idx, entry in enumerate(results, start=1):
        entry['trade_num'] = idx
    
    return results