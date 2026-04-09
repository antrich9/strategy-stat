import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    entries = []
    trade_num = 1
    
    # State variables
    session_high = np.nan
    session_low = np.nan
    first_sweep_triggered = False
    trade_executed = False
    is_long_trade = False
    is_short_trade = False
    fvg_detected = False
    bullish_fvg_high = np.nan
    bullish_fvg_low = np.nan
    bearish_fvg_high = np.nan
    bearish_fvg_low = np.nan
    long_crossed_under = False
    short_crossed_over = False
    bull_ms_count = 0
    bear_ms_count = 0
    os = 0
    upper_value = np.nan
    upper_loc = 0
    lower_value = np.nan
    lower_loc = 0
    
    length = 5
    p = length // 2
    
    last_day = -1
    
    for i in range(len(df)):
        ts = df['time'].iloc[i]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        
        new_day = dt.day != last_day
        if new_day:
            last_day = dt.day
            session_high = np.nan
            session_low = np.nan
            first_sweep_triggered = False
            trade_executed = False
            is_long_trade = False
            is_short_trade = False
            fvg_detected = False
            bullish_fvg_high = np.nan
            bullish_fvg_low = np.nan
            bearish_fvg_high = np.nan
            bearish_fvg_low = np.nan
            long_crossed_under = False
            short_crossed_over = False
            bull_ms_count = 0
            bear_ms_count = 0
            os = 0
        
        is_london = 3 <= hour < 7
        is_newyork = 8 <= hour < 12
        is_trading_window = is_london or is_newyork
        
        pdh = df['high'].iloc[i]
        pdl = df['low'].iloc[i]
        
        high_price = df['high'].iloc[i]
        low_price = df['low'].iloc[i]
        close_price = df['close'].iloc[i]
        
        if is_trading_window:
            session_high = high_price if np.isnan(session_high) else max(session_high, high_price)
            session_low = low_price if np.isnan(session_low) else min(session_low, low_price)
        
        if not first_sweep_triggered:
            if high_price >= pdh and is_trading_window:
                first_sweep_triggered = True
                is_short_trade = True
                is_long_trade = False
            elif low_price <= pdl and is_trading_window:
                first_sweep_triggered = True
                is_long_trade = True
                is_short_trade = False
        
        bfvg = low_price > df['high'].iloc[i-2] if i >= 2 else False
        sfvg = high_price < df['low'].iloc[i-2] if i >= 2 else False
        
        if not fvg_detected and first_sweep_triggered:
            if bfvg and is_long_trade:
                bullish_fvg_high = low_price
                bullish_fvg_low = df['low'].iloc[i-2] if i >= 2 else np.nan
                fvg_detected = True
            
            if sfvg and is_short_trade:
                bearish_fvg_high = high_price
                bearish_fvg_low = df['low'].iloc[i-2] if i >= 2 else np.nan
                fvg_detected = True
        
        if i >= 2 and low_price < bullish_fvg_high and not long_crossed_under:
            long_crossed_under = True
        
        if i >= 2 and high_price > bearish_fvg_high and not short_crossed_over:
            short_crossed_over = True
        
        # Market structure calculation
        if i >= p:
            dh = sum(1 if high_price > df['high'].iloc[i-1] else -1 for _ in range(p))
            dl = sum(1 if low_price > df['low'].iloc[i-1] else -1 for _ in range(p))
            
            bull_fractal = dh == -p
            bear_fractal = dl == p
            
            if bull_fractal:
                upper_value = high_price
                upper_loc = i
            
            if bear_fractal:
                lower_value = low_price
                lower_loc = i
            
            if bull_fractal and upper_value and close_price > upper_value:
                bull_ms_count += 1
                os = 1
            elif bear_fractal and lower_value and close_price < lower_value:
                bear_ms_count += 1
                os = -1
        
        # Entry conditions
        long_condition = (not trade_executed and is_trading_window and 
                         low_price < bullish_fvg_high and 
                         os == 1 and long_crossed_under)
        short_condition = (not trade_executed and is_trading_window and 
                          high_price > bearish_fvg_high and 
                          os == -1 and short_crossed_over)
        
        if long_condition:
            trade_executed = True
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': dt.isoformat(),
                'entry_price': close_price,
                'stop_loss': session_low
            })
            trade_num += 1
        
        if short_condition:
            trade_executed = True
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': dt.isoformat(),
                'entry_price': close_price,
                'stop_loss': session_high
            })
            trade_num += 1
    
    return entries