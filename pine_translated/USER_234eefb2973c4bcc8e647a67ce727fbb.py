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
    
    london_start_morning_hour = 8
    london_start_morning_minute = 0
    london_end_morning_hour = 9
    london_end_morning_minute = 55
    
    london_start_afternoon_hour = 14
    london_start_afternoon_minute = 0
    london_end_afternoon_hour = 16
    london_end_afternoon_minute = 55
    
    def is_within_time_window(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        current_minutes = hour * 60 + minute
        
        morning_start = london_start_morning_hour * 60 + london_start_morning_minute
        morning_end = london_end_morning_hour * 60 + london_end_morning_minute
        
        afternoon_start = london_start_afternoon_hour * 60 + london_start_afternoon_minute
        afternoon_end = london_end_afternoon_hour * 60 + london_end_afternoon_minute
        
        is_morning = morning_start <= current_minutes < morning_end
        is_afternoon = afternoon_start <= current_minutes < afternoon_end
        
        return is_morning or is_afternoon
    
    def wilder_rsi(prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def wilder_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    entries = []
    trade_num = 1
    
    high = df['high']
    low = df['low']
    close = df['close']
    time = df['time']
    
    in_trading_window = time.apply(is_within_time_window)
    
    bull_fvg_active = pd.Series(False, index=df.index)
    bull_fvg_top = pd.Series(np.nan, index=df.index)
    bull_fvg_bottom = pd.Series(np.nan, index=df.index)
    bull_fvg_time = pd.Series(0, index=df.index)
    
    bear_fvg_active = pd.Series(False, index=df.index)
    bear_fvg_top = pd.Series(np.nan, index=df.index)
    bear_fvg_bottom = pd.Series(np.nan, index=df.index)
    bear_fvg_time = pd.Series(0, index=df.index)
    
    active_bull_fvgs = []
    active_bear_fvgs = []
    
    for i in range(2, len(df)):
        if i < 2:
            continue
            
        x = 0
        if low.iloc[i-2] >= high.iloc[i]:
            x = -1
        elif low.iloc[i] >= high.iloc[i-2]:
            x = 1
        
        fvg_top = np.nan
        fvg_bottom = np.nan
        fvg_time = time.iloc[i]
        
        if x > 0:
            fvg_bottom = low.iloc[i]
            fvg_top = high.iloc[i-2]
        elif x < 0:
            fvg_bottom = low.iloc[i-2]
            fvg_top = high.iloc[i]
        
        if x != 0:
            fvg_dict = {
                'top': fvg_top,
                'bottom': fvg_bottom,
                'time': fvg_time,
                'bar_index': i,
                'broken': False
            }
            
            if x > 0:
                active_bull_fvgs.append(fvg_dict)
            else:
                active_bear_fvgs.append(fvg_dict)
        
        for fvg in active_bull_fvgs:
            if not fvg['broken']:
                if low.iloc[i] < fvg['bottom']:
                    fvg['broken'] = True
        
        for fvg in active_bull_fvgs:
            if not fvg['broken']:
                if low.iloc[i] < fvg['top'] and low.iloc[i] >= fvg['bottom']:
                    entry_price = df['close'].iloc[i]
                    
                    if in_trading_window.iloc[i] and not pd.isna(entry_price):
                        entry_ts = int(time.iloc[i])
                        entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                        
                        entries.append({
                            'trade_num': trade_num,
                            'direction': 'long',
                            'entry_ts': entry_ts,
                            'entry_time': entry_time_str,
                            'entry_price_guess': float(entry_price),
                            'exit_ts': 0,
                            'exit_time': '',
                            'exit_price_guess': 0.0,
                            'raw_price_a': float(entry_price),
                            'raw_price_b': float(entry_price)
                        })
                        trade_num += 1
                    
                    fvg['broken'] = True
        
        for fvg in active_bear_fvgs:
            if not fvg['broken']:
                if high.iloc[i] > fvg['top']:
                    fvg['broken'] = True
        
        for fvg in active_bear_fvgs:
            if not fvg['broken']:
                if high.iloc[i] > fvg['bottom'] and high.iloc[i] <= fvg['top']:
                    entry_price = df['close'].iloc[i]
                    
                    if in_trading_window.iloc[i] and not pd.isna(entry_price):
                        entry_ts = int(time.iloc[i])
                        entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                        
                        entries.append({
                            'trade_num': trade_num,
                            'direction': 'short',
                            'entry_ts': entry_ts,
                            'entry_time': entry_time_str,
                            'entry_price_guess': float(entry_price),
                            'exit_ts': 0,
                            'exit_time': '',
                            'exit_price_guess': 0.0,
                            'raw_price_a': float(entry_price),
                            'raw_price_b': float(entry_price)
                        })
                        trade_num += 1
                    
                    fvg['broken'] = True
        
        active_bull_fvgs = [f for f in active_bull_fvgs if not f['broken']]
        active_bear_fvgs = [f for f in active_bear_fvgs if not f['broken']]
    
    return entries