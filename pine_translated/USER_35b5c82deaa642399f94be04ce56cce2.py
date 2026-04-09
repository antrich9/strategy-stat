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
    close = df['close']
    open_col = df['open']
    high = df['high']
    low = df['low']
    
    # Parameters
    length = 20
    mult = 2.0
    macd_fast_length = 12
    macd_slow_length = 26
    macd_signal_length = 9
    rsi_length = 14
    ema_length = 200
    
    # Calculate basis (SMA)
    basis = close.rolling(length).mean()
    
    # Calculate bands
    std_dev = close.rolling(length).std()
    upper_band = basis + mult * std_dev
    lower_band = basis - mult * std_dev
    
    # Calculate EMA for trend filter
    ema200 = close.ewm(span=ema_length, adjust=False).mean()
    
    # Calculate MACD
    ema_fast = close.ewm(span=macd_fast_length, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow_length, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal_length, adjust=False).mean()
    macd_filter = macd_line > signal_line
    
    # RSI (Wilder) - not used in entry conditions but calculated for completeness
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/rsi_length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_length, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Entry conditions
    # Long: close crosses above lower_band AND macd_filter AND close > ema200
    # Short: close crosses below upper_band AND NOT macd_filter AND close < ema200
    
    close_above_lower = close > lower_band
    close_below_upper = close < upper_band
    
    close_prev_below_lower = close.shift(1) <= lower_band.shift(1)
    close_prev_above_upper = close.shift(1) >= upper_band.shift(1)
    
    long_condition = close_above_lower & close_prev_below_lower & macd_filter & (close > ema200)
    short_condition = close_below_upper & close_prev_above_upper & (~macd_filter) & (close < ema200)
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(basis.iloc[i]) or pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]) or pd.isna(ema200.iloc[i]) or pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i]):
            continue
        
        entry_price = close.iloc[i]
        ts = int(df['time'].iloc[i])
        
        if long_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries