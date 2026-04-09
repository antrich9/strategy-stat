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
    results = []
    trade_num = 1
    
    high = df['high']
    low = df['low']
    close = df['close']
    time = df['time']
    
    # Implement Wilder ATR manually
    def wilder_atr(high_series, low_series, close_series, length):
        tr1 = high_series - low_series
        tr2 = np.abs(high_series - close_series.shift(1))
        tr3 = np.abs(low_series - close_series.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = pd.Series(np.nan, index=tr.index)
        atr.iloc[length-1] = tr.iloc[:length].mean()
        
        for i in range(length, len(tr)):
            atr.iloc[i] = (atr.iloc[i-1] * (length - 1) + tr.iloc[i]) / length
        
        return atr
    
    atr_144 = wilder_atr(high, low, close, 144)
    
    # FVG detection conditions
    bullG = low > high.shift(1)
    bearG = high < low.shift(1)
    
    atr_filter = atr_144 * 0.5  # fvgTH = 0.5
    
    # Bullish FVG condition
    bull = ((low - high.shift(2)) > atr_filter) & (low > high.shift(2)) & (close.shift(1) > high.shift(2)) & ~(bullG | bullG.shift(1))
    
    # Bearish FVG condition
    bear = ((low.shift(2) - high) > atr_filter) & (high < low.shift(2)) & (close.shift(1) < low.shift(2)) & ~(bearG | bearG.shift(1))
    
    # Track FVG state
    bullFvgUpper = pd.Series(np.nan, index=df.index)
    bullFvgLower = pd.Series(np.nan, index=df.index)
    bearFvgUpper = pd.Series(np.nan, index=df.index)
    bearFvgLower = pd.Series(np.nan, index=df.index)
    last = pd.Series(np.nan, index=df.index)
    fvg_active = pd.Series(False, index=df.index)
    
    bullstop = pd.Series(np.nan, index=df.index)
    bearstop = pd.Series(np.nan, index=df.index)
    
    # Mitigation percentage tracking
    lastPct = pd.Series(0.0, index=df.index)
    
    for i in range(3, len(df)):
        if bull.iloc[i]:
            bullFvgUpper.iloc[i] = high.iloc[i-2]
            bullFvgLower.iloc[i] = low.iloc[i]
            bearFvgUpper.iloc[i] = np.nan
            bearFvgLower.iloc[i] = np.nan
            bullstop.iloc[i] = low.iloc[i-2]
            last.iloc[i] = True
            fvg_active.iloc[i] = True
        elif bear.iloc[i]:
            bearFvgUpper.iloc[i] = high.iloc[i]
            bearFvgLower.iloc[i] = low.iloc[i-2]
            bullFvgUpper.iloc[i] = np.nan
            bullFvgLower.iloc[i] = np.nan
            bearstop.iloc[i] = high.iloc[i-2]
            last.iloc[i] = False
            fvg_active.iloc[i] = True
        else:
            if i > 0:
                bullFvgUpper.iloc[i] = bullFvgUpper.iloc[i-1]
                bullFvgLower.iloc[i] = bullFvgLower.iloc[i-1]
                bearFvgUpper.iloc[i] = bearFvgUpper.iloc[i-1]
                bearFvgLower.iloc[i] = bearFvgLower.iloc[i-1]
                last.iloc[i] = last.iloc[i-1]
                bullstop.iloc[i] = bullstop.iloc[i-1]
                bearstop.iloc[i] = bearstop.iloc[i-1]
                fvg_active.iloc[i] = fvg_active.iloc[i-1]
        
        if bullG.iloc[i] or bearG.iloc[i]:
            fvg_active.iloc[i] = False
    
    # Calculate midpoints
    bullMidpoint = (bullFvgUpper + bullFvgLower) / 2
    bearMidpoint = (bearFvgUpper + bearFvgLower) / 2
    
    # Convert timestamps to datetime for time window filtering
    timestamps = pd.to_datetime(time, unit='s', utc=True)
    
    def in_london_window(ts):
        if pd.isna(ts):
            return False
        london_tz = timezone.utc
        dt = ts.tz_convert(london_tz) if ts.tzinfo else ts.replace(tzinfo=london_tz)
        hour = dt.hour
        minute = dt.minute
        total_minutes = hour * 60 + minute
        morning_start = 7 * 60 + 45
        morning_end = 9 * 60 + 45
        afternoon_start = 14 * 60 + 45
        afternoon_end = 16 * 60 + 45
        return (morning_start <= total_minutes < morning_end) or (afternoon_start <= total_minutes < afternoon_end)
    
    in_trading_window = timestamps.apply(in_london_window)
    
    # Crossunder and crossover conditions
    crossunder_bull = (low.shift(1) > bullMidpoint.shift(1)) & (low <= bullMidpoint)
    crossover_bear = (high.shift(1) < bearMidpoint.shift(1)) & (high >= bearMidpoint)
    
    # Entry conditions
    long_condition = (last == True) & crossunder_bull & in_trading_window & fvg_active & (lastPct > 0.01) & (lastPct <= 1)
    short_condition = (last == False) & crossover_bear & in_trading_window & fvg_active & (lastPct > 0.01) & (lastPct <= 1)
    
    # Generate entries
    for i in range(len(df)):
        if i > 0 and fvg_active.iloc[i]:
            prev_upper = bullFvgUpper.iloc[i-1] if not pd.isna(bullFvgUpper.iloc[i-1]) else bearFvgUpper.iloc[i-1]
            prev_lower = bullFvgLower.iloc[i-1] if not pd.isna(bullFvgLower.iloc[i-1]) else bearFvgLower.iloc[i-1]
            
            if not pd.isna(prev_upper) and not pd.isna(prev_lower) and prev_upper != prev_lower:
                current_pct = lastPct.iloc[i]
                if prev_upper > prev_lower:
                    current_pct = (prev_upper - prev_lower) / (prev_upper - prev_lower)
                lastPct.iloc[i] = current_pct
        
        if long_condition.iloc[i] and not pd.isna(bullMidpoint.iloc[i]):
            entry_ts = int(time.iloc[i])
            entry_price = bullMidpoint.iloc[i]
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if short_condition.iloc[i] and not pd.isna(bearMidpoint.iloc[i]):
            entry_ts = int(time.iloc[i])
            entry_price = bearMidpoint.iloc[i]
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return results