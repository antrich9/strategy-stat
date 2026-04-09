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
    
    # Default input values from Pine Script
    atrLength = 14
    atrMultiplier = 1.0
    minRR = 1.0
    acceptInverseRR = False
    useRSI = False
    rsiLength = 14
    rsiOverbought = 70.0
    rsiOversold = 30.0
    targetFrontRun = 2
    mintick = 0.01  # Assuming 2 decimal places for forex, adjust as needed
    
    # Calculate pips to price
    def pips_to_price(pips):
        return mintick * pips * 10
    
    n = len(df)
    if n < 2:
        return []
    
    # =========================================================================
    # INDICATORS
    # =========================================================================
    
    # Wilder's ATR
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.zeros(n)
    atr[atrLength] = np.mean(tr[1:atrLength+1])
    alpha = 1.0 / atrLength
    for i in range(atrLength + 1, n):
        atr[i] = tr[i] * alpha + atr[i-1] * (1 - alpha)
    
    # Wilder's RSI
    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gains = np.zeros(n)
    avg_losses = np.zeros(n)
    rsi_values = np.full(n, np.nan)
    
    if n > rsiLength:
        avg_gains[rsiLength] = np.mean(gains[1:rsiLength+1])
        avg_losses[rsiLength] = np.mean(losses[1:rsiLength+1])
        
        if avg_losses[rsiLength] == 0:
            rsi_values[rsiLength] = 100
        else:
            rs = avg_gains[rsiLength] / avg_losses[rsiLength]
            rsi_values[rsiLength] = 100 - (100 / (1 + rs))
        
        for i in range(rsiLength + 1, n):
            avg_gains[i] = (avg_gains[i-1] * (rsiLength - 1) + gains[i]) / rsiLength
            avg_losses[i] = (avg_losses[i-1] * (rsiLength - 1) + losses[i]) / rsiLength
            
            if avg_losses[i] == 0:
                rsi_values[i] = 100
            else:
                rs = avg_gains[i] / avg_losses[i]
                rsi_values[i] = 100 - (100 / (1 + rs))
    
    # =========================================================================
    # STRUCTURE IDENTIFICATION
    # =========================================================================
    
    # Detect structure patterns
    lower_low = np.zeros(n, dtype=bool)
    lower_close = np.zeros(n, dtype=bool)
    lower_low_lower_close = np.zeros(n, dtype=bool)
    
    higher_high = np.zeros(n, dtype=bool)
    higher_close = np.zeros(n, dtype=bool)
    higher_high_higher_close = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        lower_low[i] = low[i] < low[i-1]
        lower_close[i] = close[i] < close[i-1]
        lower_low_lower_close[i] = lower_low[i] and lower_close[i]
        
        higher_high[i] = high[i] > high[i-1]
        higher_close[i] = close[i] > close[i-1]
        higher_high_higher_close[i] = higher_high[i] and higher_close[i]
    
    # Track structure levels
    new_structure_low = np.full(n, np.nan)
    prev_structure_low = np.full(n, np.nan)
    outside_return_high = np.full(n, np.nan)
    new_structure_high = np.full(n, np.nan)
    prev_structure_high = np.full(n, np.nan)
    outside_return_low = np.full(n, np.nan)
    bearish_trend = np.zeros(n, dtype=bool)
    bullish_trend = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if lower_low_lower_close[i]:
            bearish_trend[i] = True
            bullish_trend[i] = False
            new_structure_low[i] = low[i]
            prev_structure_low[i] = low[i-1]
            if i >= 2:
                outside_return_high[i] = high[i-2]
        
        if higher_high_higher_close[i]:
            bullish_trend[i] = True
            bearish_trend[i] = False
            new_structure_high[i] = high[i]
            prev_structure_high[i] = high[i-1]
            if i >= 2:
                outside_return_low[i] = low[i-2]
    
    # Propagate trend state
    for i in range(1, n):
        if not bullish_trend[i] and not bearish_trend[i]:
            if i > 0:
                bullish_trend[i] = bullish_trend[i-1]
                bearish_trend[i] = bearish_trend[i-1]
    
    # =========================================================================
    # KILL ZONE CALCULATION
    # =========================================================================
    
    kill_zone_top = np.full(n, np.nan)
    kill_zone_bottom = np.full(n, np.nan)
    stop_level = np.full(n, np.nan)
    target_level = np.full(n, np.nan)
    
    for i in range(2, n):
        if bearish_trend[i] and not np.isnan(prev_structure_low[i]) and not np.isnan(outside_return_high[i]):
            initial_kz_top = outside_return_high[i]
            initial_kz_bottom = prev_structure_low[i]
            
            stop_level[i] = outside_return_high[i] + (atr[i] * atrMultiplier)
            target_level[i] = new_structure_low[i] - pips_to_price(targetFrontRun)
            
            stop_to_target = abs(stop_level[i] - target_level[i])
            fifty_percent_level = target_level[i] + (stop_to_target * 0.5)
            
            if acceptInverseRR:
                kill_zone_top[i] = initial_kz_top
            else:
                kill_zone_top[i] = min(fifty_percent_level, initial_kz_top)
            
            kill_zone_bottom[i] = initial_kz_bottom
        
        if bullish_trend[i] and not np.isnan(prev_structure_high[i]) and not np.isnan(outside_return_low[i]):
            initial_kz_top = prev_structure_high[i]
            initial_kz_bottom = outside_return_low[i]
            
            stop_level[i] = outside_return_low[i] - (atr[i] * atrMultiplier)
            target_level[i] = new_structure_high[i] + pips_to_price(targetFrontRun)
            
            stop_to_target = abs(target_level[i] - stop_level[i])
            fifty_percent_level = stop_level[i] + (stop_to_target * 0.5)
            
            if acceptInverseRR:
                kill_zone_bottom[i] = initial_kz_bottom
            else:
                kill_zone_bottom[i] = max(fifty_percent_level, initial_kz_bottom)
            
            kill_zone_top[i] = initial_kz_top
    
    # =========================================================================
    # ENTRY CONDITIONS
    # =========================================================================
    
    # Kill zone check
    in_kill_zone = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isnan(kill_zone_top[i]) and not np.isnan(kill_zone_bottom[i]):
            in_kill_zone[i] = close[i] >= kill_zone_bottom[i] and close[i] <= kill_zone_top[i]
    
    # RSI condition
    rsi_condition_met = np.ones(n, dtype=bool)
    if useRSI:
        for i in range(n):
            if bearish_trend[i]:
                rsi_condition_met[i] = rsi_values[i] >= rsiOverbought
            if bullish_trend[i]:
                rsi_condition_met[i] = rsi_values[i] <= rsiOversold
    
    # Reversal candle detection
    reversal_candle = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if bearish_trend[i] and in_kill_zone[i] and rsi_condition_met[i]:
            high_in_zone = high[i] >= kill_zone_bottom[i] and high[i] <= kill_zone_top[i]
            close_in_zone = close[i] >= kill_zone_bottom[i] and close[i] <= kill_zone_top[i]
            reversal_candle[i] = lower_low_lower_close[i] and high_in_zone and close_in_zone
        
        if bullish_trend[i] and in_kill_zone[i] and rsi_condition_met[i]:
            low_in_zone = low[i] >= kill_zone_bottom[i] and low[i] <= kill_zone_top[i]
            close_in_zone = close[i] >= kill_zone_bottom[i] and close[i] <= kill_zone_top[i]
            reversal_candle[i] = higher_high_higher_close[i] and low_in_zone and close_in_zone
    
    # Valid RR check
    valid_rr = np.ones(n, dtype=bool)
    for i in range(n):
        if not acceptInverseRR and not np.isnan(stop_level[i]) and not np.isnan(target_level[i]):
            risk = abs(close[i] - stop_level[i])
            reward = abs(target_level[i] - close[i])
            if risk > 0:
                actual_rr = reward / risk
                valid_rr[i] = actual_rr >= minRR
    
    # Entry conditions
    long_condition = np.zeros(n, dtype=bool)
    short_condition = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if not np.isnan(stop_level[i]) and not np.isnan(target_level[i]):
            long_condition[i] = bullish_trend[i] and reversal_candle[i] and valid_rr[i]
            short_condition[i] = bearish_trend[i] and reversal_candle[i] and valid_rr[i]
    
    # =========================================================================
    # EXECUTE ENTRIES
    # =========================================================================
    
    entries = []
    trade_num = 0
    
    for i in range(n):
        # Skip bars where required indicators are NaN
        if np.isnan(atr[i]):
            continue
        if np.isnan(stop_level[i]) or np.isnan(target_level[i]):
            continue
        
        # Long entry
        if long_condition[i]:
            trade_num += 1
            entry_price = close[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
        
        # Short entry
        if short_condition[i]:
            trade_num += 1
            entry_price = close[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
    
    return entries