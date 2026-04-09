import pandas as pd
import numpy as np
from datetime import datetime, timezone

def calculate_wilder_rsi(prices, length):
    """Wilder RSI implementation"""
    deltas = prices.diff()
    gains = deltas.where(deltas > 0, 0.0)
    losses = -deltas.where(deltas < 0, 0.0)
    avg_gain = gains.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_adx(high, low, close, length):
    """Calculate ADX, DI+, DI- using Wilder's method"""
    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)
    for i in range(1, len(high)):
        up_move = high.iloc[i] - high.iloc[i-1]
        down_move = low.iloc[i-1] - low.iloc[i]
        if up_move > down_move and up_move > 0:
            plus_dm.iloc[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm.iloc[i] = down_move
    plus_dm_sum = plus_dm.rolling(length).sum()
    minus_dm_sum = minus_dm.rolling(length).sum()
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/length, adjust=False).mean()
    di_plus = 100 * plus_dm_sum / atr
    di_minus = 100 * minus_dm_sum / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = dx.replace([np.inf, -np.inf], 0)
    adx = dx.ewm(alpha=1.0/length, adjust=False).mean()
    return di_plus, di_minus, adx

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
    n = len(df)
    adx_len = 14
    adx_threshold = 25.0

    _, _, adx = calculate_adx(df['high'], df['low'], df['close'], adx_len)

    london_utc_offset = 1
    london_start_morning_hour = 8
    london_end_morning_hour = 9
    london_start_afternoon_hour = 14
    london_end_afternoon_hour = 16

    hours = pd.Series([datetime.fromtimestamp(ts, tz=timezone.utc).hour for ts in df['time']], index=df.index)
    minutes = pd.Series([datetime.fromtimestamp(ts, tz=timezone.utc).minute for ts in df['time']], index=df.index)
    is_morning_window = (hours == london_start_morning_hour) | ((hours == london_end_morning_hour) & (minutes < 55))
    is_afternoon_window = (hours == london_start_afternoon_hour) | ((hours == london_end_afternoon_hour) & (minutes < 55))
    is_within_time_window = is_morning_window | is_afternoon_window

    ema_fast = df['close'].ewm(span=10, adjust=False).mean()
    ema_slow = df['close'].ewm(span=50, adjust=False).mean()
    ema_bullish = ema_fast > ema_slow
    ema_bearish = ema_fast < ema_slow

    fvg_bullish = (df['low'].shift(2) > df['high']) & ~df['low'].isna() & ~df['high'].shift(2).isna()
    fvg_bearish = (df['high'].shift(2) < df['low']) & ~df['high'].isna() & ~df['low'].shift(2).isna()

    bull_condition = is_within_time_window & (adx > adx_threshold) & ema_bullish & fvg_bullish
    bear_condition = is_within_time_window & (adx > adx_threshold) & ema_bearish & fvg_bearish

    for i in range(n):
        if i < 0 or i >= n:
            continue
        if pd.isna(adx.iloc[i]) or pd.isna(ema_fast.iloc[i]) or pd.isna(ema_slow.iloc[i]):
            continue
        if bull_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif bear_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1

    return entries