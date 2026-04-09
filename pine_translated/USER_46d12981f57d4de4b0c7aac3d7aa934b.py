import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    lookback_bars = 12
    threshold = 0.0
    atr_length = 14
    
    entries = []
    trade_num = 1
    
    # ATR (Wilder's method)
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift(1))
    tr3 = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_length, adjust=False).mean()
    
    # FVG detection
    low_2 = df['low'].shift(2)
    high_2 = df['high'].shift(2)
    close_1 = df['close'].shift(1)
    bull_fvg = (df['high'] < low_2) & (close_1 < low_2)
    bear_fvg = (df['low'] > high_2) & (close_1 > high_2)
    
    # Bars since functions
    def bars_since(cond):
        result = pd.Series(np.nan, index=df.index)
        count = 0
        for i in range(len(cond)):
            if cond.iloc[i]:
                count = 0
            else:
                count += 1
            result.iloc[i] = count if count > 0 else np.nan
        return result
    
    bull_since = bars_since(bear_fvg)
    bear_since = bars_since(bull_fvg)
    
    # BPR conditions
    bull_cond_1 = bull_fvg & (bull_since <= lookback_bars)
    combined_low_bull = np.where(bull_cond_1, np.maximum(df['high'].shift(bull_since.astype(int)), high_2), np.nan)
    combined_high_bull = np.where(bull_cond_1, np.minimum(df['low'].shift((bull_since + 2).astype(int)), df['low']), np.nan)
    bull_result = bull_cond_1 & ((combined_high_bull - combined_low_bull) >= threshold)
    
    bear_cond_1 = bear_fvg & (bear_since <= lookback_bars)
    combined_high_bear = np.where(bear_cond_1, np.minimum(df['low'].shift(bear_since.astype(int)), low_2), np.nan)
    combined_low_bear = np.where(bear_cond_1, np.maximum(df['high'].shift((bear_since + 2).astype(int)), df['high']), np.nan)
    bear_result = bear_cond_1 & ((combined_high_bear - combined_low_bear) >= threshold)
    
    # State tracking arrays (simulate var arrays)
    bull_boxes_low = []
    bull_boxes_high = []
    bull_labeled = []
    bear_boxes_low = []
    bear_boxes_high = []
    bear_labeled = []
    last_entry_type = None
    
    for i in range(len(df)):
        bull_entry = None
        bear_entry = None
        
        # Check for bullish entries
        new_bull_boxes_low = []
        new_bull_boxes_high = []
        new_bull_labeled = []
        for j, (box_low, box_high, labeled) in enumerate(zip(bull_boxes_low, bull_boxes_high, bull_labeled)):
            if not np.isnan(box_low):
                if df['low'].iloc[i] < box_high and not labeled:
                    bull_entry = "Entered Bullish FVG"
                    new_bull_labeled.append(True)
                else:
                    new_bull_labeled.append(labeled)
                new_bull_boxes_low.append(box_low)
                new_bull_boxes_high.append(box_high)
        
        bull_boxes_low = new_bull_boxes_low
        bull_boxes_high = new_bull_boxes_high
        bull_labeled = new_bull_labeled
        
        # Check for bearish entries
        new_bear_boxes_low = []
        new_bear_boxes_high = []
        new_bear_labeled = []
        for j, (box_low, box_high, labeled) in enumerate(zip(bear_boxes_low, bear_boxes_high, bear_labeled)):
            if not np.isnan(box_low):
                if df['high'].iloc[i] > box_low and not labeled:
                    bear_entry = "Entered Bearish FVG"
                    new_bear_labeled.append(True)
                else:
                    new_bear_labeled.append(labeled)
                new_bear_boxes_low.append(box_low)
                new_bear_boxes_high.append(box_high)
        
        bear_boxes_low = new_bear_boxes_low
        bear_boxes_high = new_bear_boxes_high
        bear_labeled = new_bear_labeled
        
        # Create new bullish box
        if bull_result.iloc[i] and not np.isnan(bull_since.iloc[i]):
            bull_boxes_low.append(df['low'].iloc[i])
            bull_boxes_high.append(combined_high_bull[i] if not np.isnan(combined_high_bull[i]) else df['high'].iloc[i])
            bull_labeled.append(False)
        
        # Create new bearish box
        if bear_result.iloc[i] and not np.isnan(bear_since.iloc[i]):
            bear_boxes_low.append(combined_low_bear[i] if not np.isnan(combined_low_bear[i]) else df['low'].iloc[i])
            bear_boxes_high.append(df['high'].iloc[i])
            bear_labeled.append(False)
        
        # Update last_entry_type
        if bull_entry:
            last_entry_type = bull_entry
        if bear_entry:
            last_entry_type = bear_entry
        
        # Long entry: consecutive bullish FVG
        if last_entry_type == "Entered Bullish FVG":
            bull_fvg_prev = bull_fvg.iloc[i-1] if i > 0 else False
            close_2 = df['close'].iloc[i-2] if i > 1 else df['close'].iloc[0]
            low_3 = df['low'].iloc[i-2] if i > 1 else df['low'].iloc[0]
            consecutive = bull_fvg_prev and close_2 < low_3
            if consecutive:
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
                trade_num += 1
        
        # Clean up NaN boxes
        bull_boxes_low = [x for x in bull_boxes_low if not np.isnan(x)]
        bull_boxes_high = [x for x in bull_boxes_high if not np.isnan(x)]
        bear_boxes_low = [x for x in bear_boxes_low if not np.isnan(x)]
        bear_boxes_high = [x for x in bear_boxes_high if not np.isnan(x)]
    
    return entries