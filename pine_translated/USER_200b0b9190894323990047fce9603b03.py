import pandas as pd
import numpy as np
from datetime import datetime, timezone

def calculate_wilder_rsi(close, period):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    for i in range(period, len(close)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)
    return rsi

def calculate_wilder_atr(high, low, close, period):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    for i in range(period, len(tr)):
        atr.iloc[i] = (atr.iloc[i-1] * (period - 1) + tr.iloc[i]) / period
    return atr

def generate_entries(df: pd.DataFrame) -> list:
    ema_length = 50
    fib_level = 0.5
    rsi_length = 14
    atr_length = 14
    swing_length = 5
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    ema = close.ewm(span=ema_length, adjust=False).mean()
    rsi = calculate_wilder_rsi(close, rsi_length)
    atr = calculate_wilder_atr(high, low, close, atr_length)
    
    df_time = pd.to_datetime(df['time'], unit='s', utc=True)
    df_4h = df.set_index(df_time).resample('4h').agg({'close': 'last'}).dropna()
    ema_4h = df_4h['close'].ewm(span=ema_length, adjust=False).mean()
    higher_tf_ema = ema_4h.reindex(df_time).ffill()
    
    rolling_high = high.rolling(window=swing_length, min_periods=swing_length).max()
    rolling_low = low.rolling(window=swing_length, min_periods=swing_length).min()
    swing_high = rolling_high.where(high == rolling_high)
    swing_low = rolling_low.where(low == rolling_low)
    swing_high = swing_high.ffill()
    swing_low = swing_low.ffill()
    
    pullback_long = swing_low + fib_level * (swing_high - swing_low)
    pullback_short = swing_high - fib_level * (swing_high - swing_low)
    
    long_condition = (close > ema) & (rsi > 30) & (close > pullback_long) & (close > higher_tf_ema)
    short_condition = (close < ema) & (rsi < 70) & (close < pullback_short) & (close < higher_tf_ema)
    
    entries = []
    trade_num = 1
    
    for i in range(ema_length, len(df)):
        if pd.isna(ema.iloc[i]) or pd.isna(rsi.iloc[i]) or pd.isna(atr.iloc[i]) or pd.isna(higher_tf_ema.iloc[i]):
            continue
        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    return entries