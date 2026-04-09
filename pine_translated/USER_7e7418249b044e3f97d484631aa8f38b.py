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
    
    def crossover(a, b):
        return (a > b) & (a.shift(1) <= b.shift(1))
    
    def crossunder(a, b):
        return (a < b) & (a.shift(1) >= b.shift(1))
    
    # Strategy parameters
    atr_length = 14
    atr_multiplier = 1.5
    gpi_period = 14
    sma_period = 5
    lookback = 20
    ret_since = 2
    ret_valid_limit = 2
    trade_direction = 'Both'
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Wilder ATR
    close_shifted = close.shift(1).fillna(method='bfill')
    tr1 = high - low
    tr2 = np.abs(high - close_shifted)
    tr3 = np.abs(low - close_shifted)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr = np.full(len(df), np.nan)
    atr[atr_length - 1] = tr.iloc[:atr_length].mean()
    for i in range(atr_length, len(df)):
        atr[i] = (atr[i - 1] * (atr_length - 1) + tr.iloc[i]) / atr_length
    atr_series = pd.Series(atr, index=df.index)
    
    # GPI (not used in entry logic but calculated for indicator parity)
    gpi_pct = ((close - close.shift(gpi_period)) / close.shift(gpi_period)) * 100
    gpi = gpi_pct.rolling(sma_period).mean()
    
    # Pivot points
    pl = low.rolling(2 * lookback + 1, min_periods=2 * lookback + 1).min()
    ph = high.rolling(2 * lookback + 1, min_periods=2 * lookback + 1).max()
    
    # Box boundaries
    s_top = pl
    s_bot_vals = np.where(low.shift(lookback + 1) > low.shift(lookback - 1),
                          low.shift(lookback - 1), low.shift(lookback + 1))
    s_bot = pd.Series(s_bot_vals, index=df.index)
    
    r_top_vals = np.where(high.shift(lookback + 1) > high.shift(lookback - 1),
                         high.shift(lookback + 1), high.shift(lookback - 1))
    r_top = pd.Series(r_top_vals, index=df.index)
    r_bot = ph
    
    # Breakout detection
    cu = crossunder(close, s_bot)  # Support breakout (bearish)
    co = crossover(close, r_top)   # Resistance breakout (bullish)
    
    # State tracking
    s_break = pd.Series(False, index=df.index)
    r_break = pd.Series(False, index=df.index)
    s_ret_valid = pd.Series(False, index=df.index)
    r_ret_valid = pd.Series(False, index=df.index)
    
    s_ret_since = np.full(len(df), -1, dtype=float)
    r_ret_since = np.full(len(df), -1, dtype=float)
    s_ret_occurred = False
    r_ret_occurred = False
    
    for i in range(len(df)):
        if i < lookback + 1:
            s_ret_since[i] = -1
            r_ret_since[i] = -1
            continue
        
        if cu.iloc[i] and not s_break.iloc[i - 1] if i > 0 else True:
            s_break.iloc[i] = True
            s_ret_since[i] = 0
            s_ret_occurred = False
        elif s_break.iloc[i - 1] if i > 0 else False:
            s_break.iloc[i] = s_break.iloc[i - 1]
            if s_ret_since[i - 1] >= 0:
                s_ret_since[i] = s_ret_since[i - 1] + 1
        
        if co.iloc[i] and not r_break.iloc[i - 1] if i > 0 else True:
            r_break.iloc[i] = True
            r_ret_since[i] = 0
            r_ret_occurred = False
        elif r_break.iloc[i - 1] if i > 0 else False:
            r_break.iloc[i] = r_break.iloc[i - 1]
            if r_ret_since[i - 1] >= 0:
                r_ret_since[i] = r_ret_since[i - 1] + 1
        
        if s_ret_since[i] >= ret_since:
            if not s_ret_occurred:
                if (high.iloc[i] >= s_top.iloc[i] and close.iloc[i] <= s_bot.iloc[i]) or \
                   (high.iloc[i] >= s_top.iloc[i] and close.iloc[i] >= s_bot.iloc[i] and close.iloc[i] <= s_top.iloc[i]) or \
                   (high.iloc[i] >= s_bot.iloc[i] and high.iloc[i] <= s_top.iloc[i]) or \
                   (high.iloc[i] >= s_bot.iloc[i] and high.iloc[i] <= s_top.iloc[i] and close.iloc[i] < s_bot.iloc[i]):
                    if s_ret_since[i] <= ret_valid_limit:
                        s_ret_valid.iloc[i] = True
                        s_ret_occurred = True
        
        if r_ret_since[i] >= ret_since:
            if not r_ret_occurred:
                if (low.iloc[i] <= r_bot.iloc[i] and close.iloc[i] >= r_top.iloc[i]) or \
                   (low.iloc[i] <= r_bot.iloc[i] and close.iloc[i] <= r_top.iloc[i] and close.iloc[i] >= r_bot.iloc[i]) or \
                   (low.iloc[i] <= r_top.iloc[i] and low.iloc[i] >= r_bot.iloc[i]) or \
                   (low.iloc[i] <= r_top.iloc[i] and low.iloc[i] >= r_bot.iloc[i] and close.iloc[i] > r_top.iloc[i]):
                    if r_ret_since[i] <= ret_valid_limit:
                        r_ret_valid.iloc[i] = True
                        r_ret_occurred = True
        
        if np.isnan(s_ret_since[i]):
            s_ret_since[i] = -1
        if np.isnan(r_ret_since[i]):
            r_ret_since[i] = -1
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if s_ret_since[i] >= 0 and s_ret_since[i] < ret_since:
            long_condition = s_ret_valid.iloc[i]
        else:
            long_condition = co.iloc[i] or s_ret_valid.iloc[i]
        
        short_condition = cu.iloc[i] or r_ret_valid.iloc[i]
        
        if long_condition and trade_direction in ['Long', 'Both']:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if short_condition and trade_direction in ['Short', 'Both']:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries