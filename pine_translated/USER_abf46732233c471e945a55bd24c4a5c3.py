import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Calculate OHLC shifted values
    # Pine Script uses [1] for previous bar, [2] for 2 bars ago, etc.
    # In pandas, shift(1) gives previous bar, shift(2) gives 2 bars ago
    
    # Calculate OB conditions
    # isUp(index) => close[index] > open[index]
    # isDown(index) => close[index] < open[index]
    
    # For isObUp(index): isDown(index+1) and isUp(index) and close[index] > high[index+1]
    # So for bar i:
    # isObUp(1) means: isDown(i) and isUp(i-1) and close[i-1] > high[i]
    # Wait, index in Pine Script is relative to current bar.
    # If index=1, it means 1 bar ago.
    # So isObUp(1) means: check 1 bar ago
    
    # Let's clarify:
    # In Pine Script:
    # isObUp(index) =>
    #     isDown(index + 1) and isUp(index) and close[index] > high[index + 1]
    # If index=1:
    #     isDown(2) and isUp(1) and close[1] > high[2]
    
    # So in Python, for bar i:
    # obUp = isDown(i-1) and isUp(i-2) and close[i-2] > high[i-1]
    # Wait, this is getting confusing.
    
    # Let's use a different approach. In Pine Script:
    # obUp = isObUp(1)
    # This is calculated once at the beginning (var variables), but for entry logic, we need it for each bar.
    
    # Actually, looking at the Pine Script, obUp, obDown, fvgUp, fvgDown are calculated once.
    # But for a trading strategy, we need these to be calculated for each bar.
    
    # Let me re-read the Pine Script:
    # obUp = isObUp(1)
    # This means: check if 1 bar ago is an OB up.
    # But this is a one-time calculation. For continuous trading, we need to check this condition on each bar.
    
    # Wait, the code says:
    # isObUp(index) =>
    #     isDown(index + 1) and isUp(index) and close[index] > high[index + 1]
    # obUp = isObUp(1)
    
    # So obUp is true if 1 bar ago (index 1) satisfies the condition:
    # - 2 bars ago was down
    # - 1 bar ago was up
    # - close of 1 bar ago > high of 2 bars ago
    
    # Similarly, fvgUp = isFvgUp(0) means:
    # isFvgUp(index) => low[index] > high[index + 2]
    # So fvgUp is true if current low > high 2 bars ago.
    
    # For entries, we need to check these conditions on each bar.
    # So for bar i (current), we need to check:
    # - Was bar i-1 an OB up? (obUp condition for bar i-1)
    # - Is bar i an FVG up? (fvgUp condition for bar i)
    
    # So the stacked condition (OB + FVG) would be:
    # Long: obUp[i-1] and fvgUp[i]
    # Short: obDown[i-1] and fvgDown[i]
    
    # But the Pine Script calculates obUp, obDown, fvgUp, fvgDown once.
    # This suggests that in the Pine Script, these are calculated on each bar, but the code shown is just the calculation part.
    
    # Given the context, I'll implement the logic to check these conditions for each bar.
    
    # Calculate indicators
    # Close, Open, High, Low
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    time = df['time']
    
    # isUp: close > open
    is_up = close > open_
    is_down = close < open_
    
    # For OB conditions, we need to check previous bars
    # isObUp(1) for bar i: isDown(i) and isUp(i-1) and close(i-1) > high(i)
    # Wait, index 1 means 1 bar ago.
    # So for bar i:
    # ob_up = isDown(i-1) and isUp(i-2) and close(i-2) > high(i-1)
    
    # Let's calculate these as Series
    
    # For bar i, obUp condition (checking if 1 bar ago is OB up):
    # isDown(i) and isUp(i-1) and close(i-1) > high(i)
    # In Python (pandas):
    # is_down shifted by 1 (i-1) AND is_up shifted by 2 (i-2) AND close shifted by 2 > high shifted by 1
    
    ob_up = is_down.shift(1) & is_up.shift(2) & (close.shift(2) > high.shift(1))
    ob_down = is_up.shift(1) & is_down.shift(2) & (close.shift(2) < low.shift(1))
    
    # For FVG:
    # isFvgUp(0): low > high[2]
    # So for bar i: low[i] > high[i+2]? No, high[2] means high 2 bars ago.
    # In Pine Script, index 0 is current, index 1 is previous, index 2 is 2 bars ago.
    # So low[0] > high[2] means current low > high 2 bars ago.
    
    fvg_up = low > high.shift(2)
    fvg_down = high < low.shift(2)
    
    # ATR calculation (Wilder's method)
    # True Range = max(high - low, abs(high - close_prev), abs(low - close_prev))
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR(20) with Wilder smoothing
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    
    # Filters (disabled by default in Pine Script)
    # inp1 = false, inp2 = false, inp3 = false
    # So volfilt = true, atrfilt = true, locfiltb = true, locfilts = true
    # But let's implement them anyway with the default values
    
    # Volume filter
    sma_vol = volume.rolling(9).mean()
    volfilt = volume.shift(1) > sma_vol * 1.5  # This is what the Pine Script does: volume[1] > ta.sma(volume, 9)*1.5
    
    # ATR filter
    # atr = ta.atr(20) / 1.5
    atr_val = atr / 1.5
    atrfilt = (low - high.shift(2) > atr_val) | (low.shift(2) - high > atr_val)
    
    # Trend filter
    loc = close.rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # BFVG and SFVG
    # bfvg = low > high[2] and volfilt and atrfilt and locfiltb
    # Note: In Pine Script, inp1, inp2, inp3 are booleans. If false, the filter is not applied (treated as true).
    # But in the code: volfilt = inp1 ? volume[1] > ta.sma(volume, 9)*1.5 : true
    # So if inp1 is false, volfilt is true.
    # Same for atrfilt and locfiltb/locfilts.
    
    # Since inp1, inp2, inp3 are inputs and default to false, we should treat them as disabled.
    # But to be thorough, I'll implement the logic as if they are disabled (i.e., always true).
    # However, the user might want the filters enabled. I'll check the Pine Script inputs.
    
    # Inputs:
    # inp1 = input.bool(false, "Volume Filter") -> so volfilt is true
    # inp2 = input.bool(false, "ATR Filter") -> so atrfilt is true
    # inp3 = input.bool(false, "Trend Filter") -> so locfiltb is true and locfilts is true
    
    # So effectively:
    bfvg = fvg_up  # low > high.shift(2)
    sfvg = fvg_down  # high < low.shift(2)
    
    # If we want to include the filters (assuming they are enabled):
    # bfvg = low > high.shift(2) & volfilt & atrfilt & locfiltb
    # sfvg = high < low.shift(2) & volfilt & atrfilt & locfilts
    
    # Time filters
    # Convert timestamp to time
    # The timestamps are UTC
    dt = pd.to_datetime(time, unit='s', utc=True)
    
    # Extract hour and minute
    hour = dt.dt.hour
    minute = dt.dt.minute
    
    # Create time as integer for comparison (e.g., 715 for 07:15)
    time_int = hour * 100 + minute
    
    # betweenTime = '0700-0959'
    # betweenTime1 = '1200-1459'
    time_filter1 = (time_int >= 700) & (time_int <= 959)
    time_filter2 = (time_int >= 1200) & (time_int <= 1459)
    time_filter = time_filter1 | time_filter2
    
    # Entry conditions
    # Long: bfvg and time_filter
    # Short: sfvg and time_filter
    
    long_condition = bfvg & time_filter
    short_condition = sfvg & time_filter
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_condition.iloc[i]:
            ts = int(time.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            ts = int(time.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
    
    return entries