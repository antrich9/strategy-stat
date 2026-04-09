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
    swing_strength = 5
    trend_filter = True
    vol_ma_period = 20
    vol_multiplier = 1.5
    require_vol_surge = True
    atr_period = 14
    atr_stop_multi = 3.0
    r_value = 1.0
    max_trades_per_day = 2
    
    n = len(df)
    if n == 0:
        return []
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    time_col = df['time']
    
    # Calculate indicators
    sma50 = close.ewm(span=50, adjust=False).mean()
    sma200 = close.ewm(span=200, adjust=False).mean()
    sma20 = close.rolling(20).mean()
    vol_ma = volume.rolling(vol_ma_period).mean()
    
    # Wilder RSI for ATR calculation
    def wilder_atr(data_high, data_low, data_close, period):
        tr1 = data_high - data_low
        tr2 = abs(data_high - data_close.shift(1))
        tr3 = abs(data_low - data_close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    atr1 = wilder_atr(high, low, close, atr_period)
    
    uptrend = sma50 > sma200
    downtrend = sma50 < sma200
    
    volume_surge = volume > vol_ma * vol_multiplier
    vol_confirmed = not require_vol_surge or volume_surge
    
    entries = []
    trade_num = 1
    trades_today = 0
    traded_this_bar = False
    last_day_ts = time_col.iloc[0] // 86400000 * 86400000
    
    for i in range(n):
        ts = int(time_col.iloc[i])
        current_day = ts // 86400000 * 86400000
        
        if current_day > last_day_ts:
            trades_today = 0
            traded_this_bar = False
            last_day_ts = current_day
        
        if i > 0:
            prev_traded = df['volume'].iloc[i] != volume.iloc[i]
        else:
            prev_traded = False
        traded_this_bar = False
        
        if pd.isna(sma20.iloc[i]) or pd.isna(sma50.iloc[i]) or pd.isna(sma200.iloc[i]) or pd.isna(atr1.iloc[i]):
            continue
        
        position_size = 0
        for entry in entries:
            if entry['entry_ts'] == ts:
                position_size = 1
                break
        
        trade_allowed = trades_today < max_trades_per_day and not traded_this_bar
        
        if position_size == 0 and trade_allowed and vol_confirmed.iloc[i]:
            if trend_filter:
                long_condition = uptrend.iloc[i] and close.iloc[i] > sma20.iloc[i]
                short_condition = downtrend.iloc[i] and close.iloc[i] < sma20.iloc[i]
            else:
                long_condition = close.iloc[i] > sma20.iloc[i]
                short_condition = close.iloc[i] < sma20.iloc[i]
            
            direction = None
            if long_condition:
                direction = 'long'
            elif short_condition:
                direction = 'short'
            
            if direction is not None:
                entry_price = close.iloc[i]
                entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
                
                entries.append({
                    'trade_num': trade_num,
                    'direction': direction,
                    'entry_ts': ts,
                    'entry_time': entry_time,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                
                trade_num += 1
                trades_today += 1
                traded_this_bar = True
    
    return entries