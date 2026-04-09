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
    
    lookback_bars = 20
    min_confidence = 30
    min_body_percent = 40
    use_liquidity_filter = True
    avoid_session_high_low = True
    check_market_structure = True
    only_trade_with_bias = True
    
    asian_high = None
    asian_low = None
    asian_open = None
    asian_close = None
    london_high = None
    london_low = None
    london_open = None
    london_close = None
    ny_open = None
    ny_high = None
    ny_low = None
    
    confidence_score = 0
    current_bias = 0
    current_direction = "NEUTRAL"
    prev_is_ny = False
    
    recent_bullish_sweep = False
    recent_bearish_sweep = False
    sweep_bar_index = 0
    recent_bull_ob = False
    recent_bear_ob = False
    ob_bar_index = 0
    last_swing_high = np.nan
    last_swing_low = np.nan
    market_structure = 0
    
    for i in range(len(df)):
        ts = df['time'].iloc[i]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        current_minute = hour * 60 + minute
        
        asian_start = 0 * 60
        asian_end = 9 * 60
        london_start = 8 * 60
        london_end = 17 * 60
        ny_start = 13 * 60
        ny_end = 22 * 60
        trading_start = 14 * 60
        trading_end = 17 * 60
        
        is_asian = asian_start <= current_minute < asian_end
        is_london = london_start <= current_minute < london_end
        is_ny = ny_start <= current_minute < ny_end
        is_trading_window = trading_start <= current_minute < trading_end
        
        if is_asian:
            if asian_high is None:
                asian_high = df['high'].iloc[i]
                asian_low = df['low'].iloc[i]
                asian_open = df['open'].iloc[i]
                asian_close = df['close'].iloc[i]
            else:
                asian_high = max(asian_high, df['high'].iloc[i])
                asian_low = min(asian_low, df['low'].iloc[i])
                asian_close = df['close'].iloc[i]
        
        if is_london:
            if london_high is None:
                london_high = df['high'].iloc[i]
                london_low = df['low'].iloc[i]
                london_open = df['open'].iloc[i]
                london_close = df['close'].iloc[i]
            else:
                london_high = max(london_high, df['high'].iloc[i])
                london_low = min(london_low, df['low'].iloc[i])
                london_close = df['close'].iloc[i]
        
        if is_ny:
            if ny_high is None:
                ny_open = df['open'].iloc[i]
                ny_high = df['high'].iloc[i]
                ny_low = df['low'].iloc[i]
            else:
                ny_high = max(ny_high, df['high'].iloc[i])
                ny_low = min(ny_low, df['low'].iloc[i])
        
        if is_ny and not prev_is_ny:
            confidence_score = 0
            
            asian_range = asian_high - asian_low if asian_high is not None and asian_low is not None else 0
            asian_momentum = asian_close - asian_open if asian_close is not None and asian_open is not None else 0
            asian_bias = 1 if asian_momentum > 0 else (-1 if asian_momentum < 0 else 0)
            
            london_range = london_high - london_low if london_high is not None and london_low is not None else 0
            london_momentum = london_close - london_open if london_close is not None and london_open is not None else 0
            london_bias = 1 if london_momentum > 0 else (-1 if london_momentum < 0 else 0)
            
            london_break_asian_high = london_high > asian_high if london_high is not None and asian_high is not None else False
            london_break_asian_low = london_low < asian_low if london_low is not None and asian_low is not None else False
            
            if asian_bias == london_bias and asian_bias != 0:
                confidence_score += asian_bias * 40
            
            if london_break_asian_high and london_bias > 0:
                confidence_score += 25
            
            if london_break_asian_low and london_bias < 0:
                confidence_score -= 25
            
            if abs(london_momentum) > london_range * 0.5:
                confidence_score += london_bias * 20
            
            if london_bias == asian_bias and london_break_asian_high and asian_bias > 0:
                confidence_score += 15
            elif london_bias == asian_bias and london_break_asian_low and asian_bias < 0:
                confidence_score -= 15
            
            if confidence_score > min_confidence:
                current_direction = "BULLISH"
                current_bias = 1
            elif confidence_score < -min_confidence:
                current_direction = "BEARISH"
                current_bias = -1
            else:
                current_direction = "NEUTRAL"
                current_bias = 0
        
        if not is_ny and prev_is_ny:
            current_direction = "NEUTRAL"
            last_bias = current_bias
            current_bias = 0
        
        prev_is_ny = is_ny
        
        swing_high = df['high'].shift(1).rolling(lookback_bars).max()
        swing_low = df['low'].shift(1).rolling(lookback_bars).min()
        
        prev_swing_high = swing_high.shift(1) if i > 0 else np.nan
        prev_swing_low = swing_low.shift(1) if i > 0 else np.nan
        
        bullish_sweep_cond = (df['low'].iloc[i] < prev_swing_low) and (df['close'].iloc[i] > prev_swing_low) if not pd.isna(prev_swing_low) else False
        bearish_sweep_cond = (df['high'].iloc[i] > prev_swing_high) and (df['close'].iloc[i] < prev_swing_high) if not pd.isna(prev_swing_high) else False
        
        if bullish_sweep_cond:
            recent_bullish_sweep = True
            recent_bearish_sweep = False
            sweep_bar_index = i
        
        if bearish_sweep_cond:
            recent_bearish_sweep = True
            recent_bullish_sweep = False
            sweep_bar_index = i
        
        if sweep_bar_index == 0:
            sweep_bar_index = i
        
        if i - sweep_bar_index > 10:
            recent_bullish_sweep = False
            recent_bearish_sweep = False
        
        if i >= 1:
            prev_body = abs(df['close'].iloc[i-1] - df['open'].iloc[i-1])
            prev_range = df['high'].iloc[i-1] - df['low'].iloc[i-1]
            prev_body_pct = (prev_body / prev_range * 100) if prev_range > 0 else 0
            
            is_bull_ob = (df['close'].iloc[i-1] < df['open'].iloc[i-1] and 
                         prev_body_pct > min_body_percent and 
                         df['close'].iloc[i] > df['high'].iloc[i-1])
            is_bear_ob = (df['close'].iloc[i-1] > df['open'].iloc[i-1] and 
                         prev_body_pct > min_body_percent and 
                         df['close'].iloc[i] < df['low'].iloc[i-1])
        else:
            is_bull_ob = False
            is_bear_ob = False
        
        if is_bull_ob:
            recent_bull_ob = True
            recent_bear_ob = False
            ob_bar_index = i
        
        if is_bear_ob:
            recent_bear_ob = True
            recent_bull_ob = False
            ob_bar_index = i
        
        if ob_bar_index == 0:
            ob_bar_index = i
        
        if i - ob_bar_index > 15:
            recent_bull_ob = False
            recent_bear_ob = False
        
        if i >= 2:
            curr_high_1 = df['high'].iloc[i-1]
            curr_low_1 = df['low'].iloc[i-1]
            
            if not pd.isna(last_swing_high):
                if curr_high_1 > last_swing_high:
                    market_structure = 1
            if curr_high_1 > last_swing_high if not pd.isna(last_swing_high) else False:
                last_swing_high = curr_high_1
            else:
                last_swing_high = curr_high_1
            
            if not pd.isna(last_swing_low):
                if curr_low_1 < last_swing_low:
                    market_structure = -1
            if curr_low_1 < last_swing_low if not pd.isna(last_swing_low) else False:
                last_swing_low = curr_low_1
            else:
                last_swing_low = curr_low_1
        
        bull_fvg = df['low'].iloc[i] > df['high'].iloc[i-2] if i >= 2 else False
        bear_fvg = df['high'].iloc[i] < df['low'].iloc[i-2] if i >= 2 else False
        
        entry_price = df['close'].iloc[i]
        
        long_conditions = [
            is_trading_window,
            current_direction == "BULLISH" if only_trade_with_bias else current_bias > 0,
            not recent_bullish_sweep if use_liquidity_filter else True,
            recent_bull_ob,
            market_structure >= 0 if check_market_structure else True,
            bull_fvg,
        ]
        
        short_conditions = [
            is_trading_window,
            current_direction == "BEARISH" if only_trade_with_bias else current_bias < 0,
            not recent_bearish_sweep if use_liquidity_filter else True,
            recent_bear_ob,
            market_structure <= 0 if check_market_structure else True,
            bear_fvg,
        ]
        
        if avoid_session_high_low:
            if asian_high is not None and entry_price >= asian_high:
                long_conditions.append(False)
            if asian_low is not None and entry_price <= asian_low:
                short_conditions.append(False)
            if london_high is not None and entry_price >= london_high:
                long_conditions.append(False)
            if london_low is not None and entry_price <= london_low:
                short_conditions.append(False)
        
        if all(long_conditions):
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
        
        if all(short_conditions):
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