import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Ensure we have enough data
    if len(df) < 5:  # Need at least 5 bars for swing detection (needs [2], [3], [4])
        return []
    
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Calculate indicators
    
    # 1. Swing Detection (not strictly needed for entries but needed for context? No, entries are based on FVG)
    # Actually, looking at the code, swing detection is used for SL/TP which we ignore, but we need the last_swing_high/low?
    # No, the entry is based purely on bfvg/sfvg.
    # But wait - the FVG condition uses high[2] and low[2], which are just shifted values.
    
    # 2. Filters
    # Volume Filter: volume[1] > sma(volume, 9) * 1.5
    sma_9 = data['volume'].shift(1).rolling(9).mean()
    # Note: volume[1] means previous bar's volume. In pandas, shift(1) gives previous value.
    # So volume[1] > sma(volume, 9) means:
    # data['volume'].shift(1) > data['volume'].rolling(9).mean()
    # But wait, ta.sma(volume, 9) is the SMA of the last 9 bars including current.
    # In Pine: ta.sma(volume, 9) at bar i = mean of volume[i-8] to volume[i]
    # So at bar i, volume[1] refers to volume[i-1].
    # So condition is: volume[i-1] > ta.sma(volume, 9)[i] * 1.5
    # Which is: data['volume'].shift(1) > data['volume'].rolling(9).mean() * 1.5
    
    vol_filt = data['volume'].shift(1) > sma_9 * 1.5
    
    # ATR Filter: atr = ta.atr(20) / 1.5
    # Then: (low - high[2] > atr) or (low[2] - high > atr)
    # Need to implement Wilder ATR
    
    # Calculate True Range components for ATR(20)
    high = data['high']
    low = data['low']
    close = data['close']
    
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder ATR(20) - uses EMA with alpha = 1/20
    atr = tr.ewm(span=20, adjust=False).mean()
    atr_val = atr / 1.5
    
    # ATR Filter condition:
    # (low - high[2] > atr) or (low[2] - high > atr)
    # low[i] - high[i-2] > atr[i]
    # low[2] means low shifted by 2
    atrfilt = (low - high.shift(2) > atr_val) | (low.shift(2) - high > atr_val)
    
    # Trend Filter
    # loc = ta.sma(close, 54)
    loc = close.rolling(54).mean()
    # loc2 = loc > loc[1]
    loc2 = loc > loc.shift(1)
    
    # locfiltb = inp3 ? loc2 : true (for bullish)
    # locfilts = inp3 ? not loc2 : true (for bearish)
    # inp3 is input.bool(false, ...), so default is false (filter disabled)
    # When disabled: locfiltb = true, locfilts = true
    # When enabled: locfiltb = loc2, locfilts = not loc2
    
    # Since inp3 defaults to false, we set:
    locfiltb = loc2  # This is actually always true if we follow the default... wait.
    # No, the Pine code is: locfiltb = inp3 ? loc2 : true
    # If inp3 is false (default), locfiltb = true
    # If inp3 is true, locfiltb = loc2
    
    # Same for locfilts: inp3 ? not loc2 : true
    # If inp3 is false (default), locfilts = true
    # If inp3 is true, locfilts = not loc2
    
    # Since inp3 is an input and defaults to false, and we don't have access to change inputs,
    # we should assume the default behavior where inp3=false.
    # So locfiltb = True and locfilts = True.
    
    # Wait, but I should probably make this configurable or just use the default.
    # The user didn't provide inputs, so I assume defaults.
    # Default inp1=false, inp2=false, inp3=false.
    # So: volfilt = true, atrfilt = true, locfiltb = true, locfilts = true.
    
    # Actually, looking at the Pine code:
    # volfilt = inp1 ? volume[1] > ta.sma(volume, 9) * 1.5 : true
    # So if inp1 is false, volfilt is true (no filter).
    
    # Let me redo the filters with the default assumption (inputs are false):
    vol_filt = True  # Always true when inp1=false
    atr_filt = True  # Always true when inp2=false
    loc_filt_bull = True  # Always true when inp3=false
    loc_filt_bear = True  # Always true when inp3=false
    
    # But the code defines them using ternary operators based on inputs.
    # Since we don't have the inputs, we should probably implement it such that
    # if the inputs were false (default), the filters are always true.
    
    # However, looking at the Pine code again:
    # volfilt = inp1 ? volume[1] > ta.sma(volume, 9) * 1.5 : true
    # This means: if inp1 is true, use the volume condition, else use true.
    
    # Since inp1, inp2, inp3 are all inputs that default to false,
    # the filters will all be True.
    
    # But to be thorough, I should probably check if the user wants me to
    # implement the filter logic or assume defaults.
    # The user said: "Use ONLY pandas and numpy"
    # and "Pine ta.ema(...)" etc.
    # It seems the user expects me to implement the logic as-is from Pine,
    # assuming default input values (false).
    
    # So:
    # volfilt = True (since inp1=false)
    # atrfilt = True (since inp2=false)
    # locfiltb = True (since inp3=false)
    # locfilts = True (since inp3=false)
    
    # But wait, the code calculates them regardless:
    # volfilt = inp1 ? ... : true
    # atrfilt = inp2 ? ... : true
    # etc.
    
    # So I should calculate them but they will be True by default.
    
    # Let me calculate them properly for the case where inputs might be true,
    # but since we don't have inputs, I'll assume they are the default values (false).
    # Actually, I should calculate the boolean conditions but they will be True when inputs are False.
    
    # Let's calculate the raw filters:
    # Volume filter condition:
    vol_cond = data['volume'].shift(1) > data['volume'].rolling(9).mean() * 1.5
    # ATR filter condition:
    atr_cond = (low - high.shift(2) > atr_val) | (low.shift(2) - high > atr_val)
    # Trend filter condition:
    loc_cond = loc > loc.shift(1)
    
    # Since inp1=inp2=inp3=False (defaults), all filters are disabled:
    # volfilt = True
    # atrfilt = True
    # locfiltb = True
    # locfilts = True
    
    # But to make the code reusable, I should probably incorporate the input checks.
    # However, since we don't have access to the inputs, I'll assume they are False.
    
    # Actually, let's look at how the FVG is defined:
    # bfvg = low > high[2] and volfilt and atrfilt and locfiltb
    # sfvg = high < low[2] and volfilt and atrfilt and locfilts
    
    # If all filters are True (disabled), then:
    # bfvg = low > high[2]
    # sfvg = high < low[2]
    
    # This makes sense for the "default" behavior.
    
    # Calculate FVG
    # Bullish FVG: low > high[2]
    # Bearish FVG: high < low[2]
    bfvg = (low > high.shift(2)) & vol_filt & atr_filt & loc_filt_bull
    sfvg = (high < low.shift(2)) & vol_filt & atr_filt & loc_filt_bear
    
    # Wait, I need to be careful about the filters. Let me recalculate with the actual conditions.
    # Actually, since inp1, inp2, inp3 default to False, the filters are effectively disabled.
    # But I should code it such that if they were enabled, it would work.
    
    # Let's code it with the assumption that the inputs are the default values (False).
    # So volfilt = True, atrfilt = True, locfiltb = True, locfilts = True.
    
    # Actually, looking at the Pine code more carefully:
    # The variables inp1, inp2, inp3 are inputs.
    # The code:
    # volfilt = inp1 ? volume[1] > ta.sma(volume, 9) * 1.5 : true
    # means if inp1 is true, use the condition, else true.
    
    # Since we don't have inp1, inp2, inp3 values, we must assume the defaults.
    # Defaults: inp1=false, inp2=false, inp3=false.
    # So: volfilt = true, atrfilt = true, locfiltb = true, locfilts = true.
    
    # Therefore:
    bfvg = (low > high.shift(2))  # & True & True & True
    sfvg = (high < low.shift(2))   # & True & True & True
    
    # But to be more accurate to the Pine code structure, I should show:
    # volfilt = True (disabled)
    # atrfilt = True (disabled)
    # locfiltb = True (disabled)
    # locfilts = True (disabled)
    
    # Generate entries
    entries = []
    trade_num = 1
    
    # Iterate through bars
    for i in range(len(data)):
        # Check for NaN in required values
        # We need close[i], low[i], high[i], and shifted values
        # Since we use shift(1), shift(2), etc., we need to handle NaN at the beginning
        # But we start from index 4 or 5 to avoid NaN from shifts, or we check for NaN.
        
        # Actually, for bfvg/sfvg, we need:
        # low[i], high[i], high[i-2], low[i-2]
        # So we need i >= 2 for the shifts to be valid (not NaN from the shift operation itself,
        # but from the data being NaN or the shift creating NaN at the start).
        # Since data is sorted oldest first, shift(2) will give NaN for first 2 rows.
        
        # Also, for ATR we use ewm which handles NaN, but the condition might be NaN.
        # We should skip bars where bfvg or sfvg is NaN or False.
        
        if pd.isna(bfvg.iloc[i]) or pd.isna(sfvg.iloc[i]):
            continue
            
        entry_price = data['close'].iloc[i]
        ts = data['time'].iloc[i]
        
        if bfvg.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
        elif sfvg.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1
    
    return entries