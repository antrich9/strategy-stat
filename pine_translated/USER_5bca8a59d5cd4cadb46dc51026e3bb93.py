import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Ensure time column is int
    df = df.copy()
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('dt')
    # Compute 4h high/low/close previous bar
    bars_4h = df[['high','low','close']].resample('4h').agg({'high':'max','low':'min','close':'last'})
    bars_4h_prev = bars_4h.shift(1)
    # Map to each row using forward fill
    df['tf4h_high'] = bars_4h_prev['high'].reindex(df.index, method='ffill')
    df['tf4h_low'] = bars_4h_prev['low'].reindex(df.index, method='ffill')
    df['tf4h_close'] = bars_4h_prev['close'].reindex(df.index, method='ffill')
    # Compute previous day high/low
    daily = df[['high','low']].resample('D').agg({'high':'max','low':'min'})
    daily_prev = daily.shift(1)
    df['prev_day_high'] = daily_prev['high'].reindex(df.index, method='ffill')
    df['prev_day_low'] = daily_prev['low'].reindex(df.index, method='ffill')
    # Reset index to have dt as column for later use
    df = df.reset_index()
    # Compute EMA 200
    ema = df['close'].ewm(span=200, adjust=False).mean()
    # Trend conditions
    in_uptrend = (df['close'] > ema) & (df['tf4h_close'] > ema)
    in_downtrend = (df['close'] < ema) & (df['tf4h_close'] < ema)
    # Liquidity sweeps
    liquidity_sweep_low = (df['low'] < df['prev_day_low']) & (df['close'] > df['prev_day_low'])
    liquidity_sweep_high = (df['high'] > df['prev_day_high']) & (df['close'] < df['prev_day_high'])
    # FVG detection
    low_prev = df['low'].shift(1)
    low_prev2 = df['low'].shift(2)
    high_prev = df['high'].shift(1)
    high_prev2 = df['high'].shift(2)
    bullish_fvg = (low_prev > high_prev2) & (df['low'] > high_prev2)
    bearish_fvg = (high_prev < low_prev2) & (df['high'] < low_prev2)
    # Order block detection
    low_rolling_min_3 = df['low'].rolling(3).min()
    ob_bull = (low_rolling_min_3 == df['low'].shift(1)) & (df['close'] > df['open'])
    high_rolling_max_3 = df['high'].rolling(3).max()
    ob_bear = (high_rolling_max_3 == df['high'].shift(1)) & (df['close'] < df['open'])
    # Friday morning filter
    df['dayofweek'] = df['dt'].dt.dayofweek
    df['hour'] = df['dt'].dt.hour
    is_friday_morning = (df['dayofweek'] == 4) & (df['hour'] < 12)
    # Entry conditions
    long_condition = in_uptrend & liquidity_sweep_low & bullish_fvg & ob_bull & ~is_friday_morning
    short_condition = in_downtrend & liquidity_sweep_high & bearish_fvg & ob_bear & ~is_friday_morning
    # Fill NaNs with False
    long_condition = long_condition.fillna(False)
    short_condition = short_condition.fillna(False)
    # Generate entries
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
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
        if short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
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
    return entries