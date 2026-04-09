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
    
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    dmi_length = 14
    adx_smoothing = 14
    cmo_period = 14
    rsi_length = 14
    stoch_length = 14
    adx_threshold = 30
    cmo_threshold = 30
    
    hl2 = (df['high'] + df['low']) / 2.0
    
    jaw = hl2.ewm(alpha=1/jaw_period, adjust=False).mean()
    teeth = hl2.ewm(alpha=1/teeth_period, adjust=False).mean()
    lips = hl2.ewm(alpha=1/lips_period, adjust=False).mean()
    
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm_series = pd.Series(plus_dm, index=df.index)
    minus_dm_series = pd.Series(minus_dm, index=df.index)
    
    atr = (df['high'] - df['low']).ewm(alpha=1/dmi_length, adjust=False).mean()
    
    plus_di = 100 * plus_dm_series.ewm(alpha=1/dmi_length, adjust=False).mean() / atr
    minus_di = 100 * minus_dm_series.ewm(alpha=1/dmi_length, adjust=False).mean() / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/adx_smoothing, adjust=False).mean()
    
    close_diff = df['close'].diff()
    gains = np.where(close_diff > 0, close_diff, 0.0)
    losses = np.where(close_diff < 0, -close_diff, 0.0)
    gains_series = pd.Series(gains, index=df.index)
    losses_series = pd.Series(losses, index=df.index)
    
    avg_gain = gains_series.ewm(alpha=1/cmo_period, adjust=False).mean()
    avg_loss = losses_series.ewm(alpha=1/cmo_period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    cmo = 100 * (rs - 1) / (rs + 1)
    
    prev_close = df['close'].shift(1)
    delta = df['close'].diff()
    
    gains_rsi = np.where(delta > 0, delta, 0.0)
    losses_rsi = np.where(delta < 0, -delta, 0.0)
    gains_rsi_series = pd.Series(gains_rsi, index=df.index)
    losses_rsi_series = pd.Series(losses_rsi, index=df.index)
    
    avg_gain_rsi = gains_rsi_series.ewm(alpha=1/rsi_length, adjust=False).mean()
    avg_loss_rsi = losses_rsi_series.ewm(alpha=1/rsi_length, adjust=False).mean()
    
    rs_rsi = avg_gain_rsi / avg_loss_rsi
    rsi_vals = 100 - (100 / (1 + rs_rsi))
    
    lowest_rsi = rsi_vals.rolling(window=stoch_length, min_periods=1).min()
    highest_rsi = rsi_vals.rolling(window=stoch_length, min_periods=1).max()
    
    stoch_rsi_raw = 100 * (rsi_vals - lowest_rsi) / (highest_rsi - lowest_rsi)
    k = stoch_rsi_raw.rolling(window=3, min_periods=1).mean()
    
    sma50 = df['close'].rolling(window=50, min_periods=1).mean()
    
    long_condition = (
        (df['close'] > jaw) & (df['close'] > teeth) & (df['close'] > lips) &
        (lips > teeth) & (teeth > jaw) &
        (adx > adx_threshold) &
        (cmo > cmo_threshold) &
        (k < 20) &
        (df['close'] > sma50)
    )
    
    short_condition = (
        (df['close'] < jaw) & (df['close'] < teeth) & (df['close'] < lips) &
        (lips < teeth) & (teeth < jaw) &
        (adx > adx_threshold) &
        (cmo < -cmo_threshold) &
        (k > 80) &
        (df['close'] < sma50)
    )
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(jaw.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        
        if short_condition.iloc[i]:
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries