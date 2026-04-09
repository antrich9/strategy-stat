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
    
    # Helper functions
    def is_up(idx):
        return df['close'].iloc[idx] > df['open'].iloc[idx]
    
    def is_down(idx):
        return df['close'].iloc[idx] < df['open'].iloc[idx]
    
    # Get previous day's high and low using 1-day offset
    # For Pine's request.security with 'D', we get previous day's H/L
    df['_day'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.date
    daily_agg = df.groupby('_day').agg({'high': 'max', 'low': 'min'}).reset_index()
    daily_agg['_day'] = pd.to_datetime(daily_agg['_day'])
    
    # Merge daily data back to df and shift by 1 day for previous day values
    df['_date'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.normalize()
    df = df.merge(daily_agg.rename(columns={'high': 'prev_day_high', 'low': 'prev_day_low', '_day': '_date'}), 
                  on='_date', how='left')
    
    # Shift to get previous day's values
    df['prev_day_high'] = df['prev_day_high'].shift(1)
    df['prev_day_low'] = df['prev_day_low'].shift(1)
    
    # Get current day's high/low using 240min (4h) aggregation
    df['_datetime'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df['_4h'] = df['_datetime'].dt.floor('4h')
    h4_agg = df.groupby('_4h').agg({'high': 'max', 'low': 'min'}).reset_index()
    h4_agg.rename(columns={'_4h': '_datetime'}, inplace=True)
    df = df.merge(h4_agg.rename(columns={'high': 'curr_4h_high', 'low': 'curr_4h_low'}), on='_datetime', how='left')
    
    # Wilder RSI implementation
    def wilder_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Wilder ATR implementation
    def wilder_atr(high, low, close, period=20):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    # Calculate indicators
    df['volume_sma9'] = df['volume'].rolling(9).mean()
    df['atr20'] = wilder_atr(df['high'], df['low'], df['close'], 20) / 1.5
    df['loc'] = df['close'].rolling(54).mean()
    df['loc_prev'] = df['loc'].shift(1)
    
    # Filter conditions (inp1=False, inp2=False, inp3=False means all True)
    volfilt = df['volume'].shift(1) > df['volume_sma9'] * 1.5
    atrfilt_cond1 = df['low'] - df['high'].shift(2) > df['atr20']
    atrfilt_cond2 = df['low'].shift(2) - df['high'] > df['atr20']
    atrfilt = atrfilt_cond1 | atrfilt_cond2
    loc_uptrend = df['loc'] > df['loc_prev']
    
    # Bullish FVG: low > high[2] (bullish fair value gap)
    # Bearish FVG: high < low[2] (bearish fair value gap)
    df['bfvg'] = (df['low'] > df['high'].shift(2)) & (volfilt) & (atrfilt) & (loc_uptrend)
    df['sfvg'] = (df['high'] < df['low'].shift(2)) & (volfilt) & (atrfilt) & (~loc_uptrend)
    
    # Order Block conditions (using shifted indices to match Pine Script)
    # isObUp: isDown(index + 1) and isUp(index) and close[index] > high[index + 1]
    # Using index i, this means: isDown(i-1) and isUp(i) and close[i] > high[i-1]
    # In shifted terms: isDown shifted by -1, isUp at 0, close > high shifted by -1
    isDown_shifted = df['close'].shift(-1) < df['open'].shift(-1)
    isUp_curr = df['close'] > df['open']
    close_above_prev_high = df['close'] > df['high'].shift(-1)
    df['obUp'] = isDown_shifted & isUp_curr & close_above_prev_high
    
    # isObDown: isUp(index + 1) and isDown(index) and close[index] < low[index + 1]
    isUp_shifted = df['close'].shift(-1) > df['open'].shift(-1)
    isDown_curr = df['close'] < df['open']
    close_below_prev_low = df['close'] < df['low'].shift(-1)
    df['obDown'] = isUp_shifted & isDown_curr & close_below_prev_low
    
    # PDHL Sweep conditions
    # Bullish: price sweeping/crossing above previous day's high
    df['pdhl_sweep_bull'] = (df['close'] > df['prev_day_high']) & (df['close'].shift(1) <= df['prev_day_high'])
    
    # FVG conditions at current bar (index 0 in Pine)
    df['fvgUp'] = df['low'] > df['high'].shift(2)
    df['fvgDown'] = df['high'] < df['low'].shift(2)
    
    # Entry conditions
    # Bullish: OB up + FVG up + PDHL sweep
    bull_entry = df['obUp'] & df['fvgUp'] & df['pdhl_sweep_bull']
    
    entries = []
    trade_num = 1
    
    n = len(df)
    for i in range(2, n):
        if bull_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries