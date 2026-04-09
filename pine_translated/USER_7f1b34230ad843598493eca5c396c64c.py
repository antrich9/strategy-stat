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
    trade_num = 0
    
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Ensure we have enough data
    if len(data) < 20:
        return results
    
    # Helper functions for OB and FVG detection
    def is_up(idx):
        return data['close'].iloc[idx] > data['open'].iloc[idx]
    
    def is_down(idx):
        return data['close'].iloc[idx] < data['open'].iloc[idx]
    
    def is_ob_up(idx):
        return is_down(idx + 1) and is_up(idx) and data['close'].iloc[idx] > data['high'].iloc[idx + 1]
    
    def is_ob_down(idx):
        return is_up(idx + 1) and is_down(idx) and data['close'].iloc[idx] < data['low'].iloc[idx + 1]
    
    def is_fvg_up(idx):
        return data['low'].iloc[idx] > data['high'].iloc[idx + 2]
    
    def is_fvg_down(idx):
        return data['high'].iloc[idx] < data['low'].iloc[idx + 2]
    
    # Calculate OB and FVG conditions
    ob_up = [False] * len(data)
    ob_down = [False] * len(data)
    fvg_up = [False] * len(data)
    fvg_down = [False] * len(data)
    
    for i in range(2, len(data) - 2):
        try:
            ob_up[i] = is_ob_up(1)
            ob_down[i] = is_ob_down(1)
            fvg_up[i] = is_fvg_up(0)
            fvg_down[i] = is_fvg_down(0)
        except:
            pass
    
    ob_up = pd.Series(ob_up)
    ob_down = pd.Series(ob_down)
    fvg_up = pd.Series(fvg_up)
    fvg_down = pd.Series(fvg_down)
    
    # Wilder RSI implementation
    def wilders_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilder ATR implementation
    def wilders_atr(high, low, close, period):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    # Calculate indicators
    volume_sma = data['volume'].rolling(9).mean()
    atr_indicator = wilders_atr(data['high'], data['low'], data['close'], 20) / 1.5
    loc = data['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    
    # Filters (assuming inp1, inp2, inp3 are True based on the strategy description)
    inp1 = True  # Volume Filter
    inp2 = True  # ATR Filter
    inp3 = True  # Trend Filter
    
    volfilt = volume_sma.shift(1) * 1.5
    volfilt = data['volume'].shift(1) > volfilt
    
    atrfilt_long = (data['low'] - data['high'].shift(2)) > atr_indicator
    atrfilt_short = (data['low'].shift(2) - data['high']) > atr_indicator
    atrfilt = atrfilt_long | atrfilt_short
    
    locfiltb = loc2
    locfilts = ~loc2
    
    # FVG conditions
    bfvg = (data['low'] > data['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (data['high'] < data['low'].shift(2)) & volfilt & atrfilt & locfilts
    
    # Previous day high and low (using daily aggregation)
    data['date'] = pd.to_datetime(data['time'], unit='s', utc=True).dt.date
    daily_agg = data.groupby('date').agg({'high': 'max', 'low': 'min'}).shift(1)
    data['prev_day_high'] = data['date'].map(daily_agg['high'])
    data['prev_day_low'] = data['date'].map(daily_agg['low'])
    
    # Trading window detection (London time: 7:45-9:45 and 14:45-16:45)
    dt = pd.to_datetime(data['time'], unit='s', utc=True)
    hours = dt.dt.hour
    minutes = dt.dt.minute
    total_minutes = hours * 60 + minutes
    
    london_start_morning = 7 * 60 + 45  # 465
    london_end_morning = 9 * 60 + 45   # 585
    london_start_afternoon = 14 * 60 + 45  # 885
    london_end_afternoon = 16 * 60 + 45    # 1005
    
    is_within_morning = (total_minutes >= london_start_morning) & (total_minutes < london_end_morning)
    is_within_afternoon = (total_minutes >= london_start_afternoon) & (total_minutes < london_end_afternoon)
    in_trading_window = is_within_morning | is_within_afternoon
    
    # Combined entry conditions
    # Long entries: bfvg OR (ob_up AND fvg_up)
    long_condition = bfvg | (ob_up & fvg_up)
    long_condition = long_condition & in_trading_window
    
    # Short entries: sfvg OR (ob_down AND fvg_down)
    short_condition = sfvg | (ob_down & fvg_down)
    short_condition = short_condition & in_trading_window
    
    # Also require that we have enough history for indicators (skip NaN)
    valid_start = 20  # ATR period
    long_condition = long_condition & (data.index >= valid_start)
    short_condition = short_condition & (data.index >= valid_start)
    
    # Filter out NaN values from indicators
    long_condition = long_condition & ~atr_indicator.isna()
    short_condition = short_condition & ~atr_indicator.isna()
    
    # Find entry signals
    long_signals = long_condition[long_condition].index.tolist()
    short_signals = short_condition[short_condition].index.tolist()
    
    # Combine and sort all signals by index
    all_signals = []
    for idx in long_signals:
        all_signals.append((idx, 'long'))
    for idx in short_signals:
        all_signals.append((idx, 'short'))
    
    all_signals.sort(key=lambda x: x[0])
    
    # Generate entry records
    for idx, direction in all_signals:
        trade_num += 1
        entry_ts = int(data['time'].iloc[idx])
        entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
        entry_price = float(data['close'].iloc[idx])
        
        results.append({
            'trade_num': trade_num,
            'direction': direction,
            'entry_ts': entry_ts,
            'entry_time': entry_time,
            'entry_price_guess': entry_price,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': entry_price,
            'raw_price_b': entry_price
        })
    
    return results