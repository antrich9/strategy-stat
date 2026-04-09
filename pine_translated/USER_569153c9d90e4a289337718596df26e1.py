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
    if len(df) < 2:
        return []

    # Default parameter values from Pine Script inputs
    donch_length = 20
    atr_period = 14
    min_body_pct = 70
    search_factor = 1.3
    ma_type = 'SMA'
    ma_length = 20
    ma_reaction = 1
    ma_type_b = 'SMA'
    ma_length_b = 8
    ma_reaction_b = 1
    filter_type = 'CON FILTRADO DE TENDENCIA'
    activate_green_elephant = True
    activate_red_elephant = True

    # Excluded time check for London 8:00-8:59 and 9:00-9:59 (Europe/London in Pine uses UTC offset)
    excluded_start_8 = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).replace(hour=4, minute=0, second=0, microsecond=0).timestamp())
    excluded_end_8 = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).replace(hour=4, minute=59, second=59, microsecond=999999).timestamp())
    excluded_start_9 = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).replace(hour=5, minute=0, second=0, microsecond=0).timestamp())
    excluded_end_9 = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).replace(hour=5, minute=59, second=59, microsecond=999999).timestamp())
    is_excluded_time = ((df['time'] >= excluded_start_8.values) & (df['time'] <= excluded_end_8.values)) | \
                       ((df['time'] >= excluded_start_9.values) & (df['time'] <= excluded_end_9.values))

    # Donchian Channel
    highest_high = df['high'].rolling(donch_length).max()
    lowest_low = df['low'].rolling(donch_length).min()
    middle_band = (highest_high + lowest_low) / 2

    # ATR Calculation (Wilder)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/atr_period, adjust=False).mean()

    # Moving Averages
    slow_ma = df['close'].rolling(ma_length).mean()
    fast_ma = df['close'].rolling(ma_length_b).mean()

    # Trend direction calculation
    slow_ma_trend = pd.Series(0, index=df.index)
    fast_ma_trend = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if pd.notna(fast_ma.iloc[i]) and pd.notna(fast_ma.iloc[i-1]):
            if fast_ma.iloc[i] >= fast_ma.iloc[i-1] + ma_reaction_b * (fast_ma.iloc[i-1] / 100):
                fast_ma_trend.iloc[i] = 1
            elif fast_ma.iloc[i] <= fast_ma.iloc[i-1] - ma_reaction_b * (fast_ma.iloc[i-1] / 100):
                fast_ma_trend.iloc[i] = -1
            else:
                fast_ma_trend.iloc[i] = fast_ma_trend.iloc[i-1]
        if pd.notna(slow_ma.iloc[i]) and pd.notna(slow_ma.iloc[i-1]):
            if slow_ma.iloc[i] >= slow_ma.iloc[i-1] + ma_reaction * (slow_ma.iloc[i-1] / 100):
                slow_ma_trend.iloc[i] = 1
            elif slow_ma.iloc[i] <= slow_ma.iloc[i-1] - ma_reaction * (slow_ma.iloc[i-1] / 100):
                slow_ma_trend.iloc[i] = -1
            else:
                slow_ma_trend.iloc[i] = slow_ma_trend.iloc[i-1]

    # Trend conditions
    is_price_above_fast_ma = (fast_ma_trend > 0) & (df['close'] > fast_ma)
    is_price_above_slow_ma = (slow_ma_trend > 0) & (df['close'] > slow_ma)
    is_price_above_both_ma = is_price_above_fast_ma & is_price_above_slow_ma
    is_slow_ma_trend_bullish = slow_ma_trend > 0
    is_fast_ma_trend_bullish = fast_ma_trend > 0
    is_both_ma_trend_bullish = is_slow_ma_trend_bullish & is_fast_ma_trend_bullish
    is_price_below_fast_ma = (fast_ma_trend < 0) & (df['close'] < fast_ma)
    is_price_below_slow_ma = (slow_ma_trend < 0) & (df['close'] < slow_ma)
    is_price_below_both_ma = is_price_below_fast_ma & is_price_below_slow_ma
    is_slow_ma_trend_bearish = slow_ma_trend < 0
    is_fast_ma_trend_bearish = fast_ma_trend < 0
    is_both_ma_trend_bearish = is_slow_ma_trend_bearish & is_fast_ma_trend_bearish

    # Elephant candle conditions
    body = np.abs(df['close'] - df['open'])
    range_hl = df['high'] - df['low']
    body_pct = body * 100 / range_hl
    is_green_elephant = df['close'] > df['open']
    is_red_elephant = df['close'] < df['open']
    is_green_valid = is_green_elephant & (body_pct >= min_body_pct)
    is_red_valid = is_red_elephant & (body_pct >= min_body_pct)
    is_green_strong = is_green_valid & (body >= atr.shift(1) * search_factor)
    is_red_strong = is_red_valid & (body >= atr.shift(1) * search_factor)

    # Final elephant conditions
    final_green_elephant = is_green_strong & (is_price_above_fast_ma | is_price_above_slow_ma | is_price_above_both_ma | is_price_above_slow_ma & is_slow_ma_trend_bullish | is_price_above_fast_ma & is_fast_ma_trend_bullish | is_price_above_both_ma & is_both_ma_trend_bullish | is_slow_ma_trend_bullish | is_fast_ma_trend_bullish | is_both_ma_trend_bullish)
    final_red_elephant = is_red_strong & (is_price_below_fast_ma | is_price_below_slow_ma | is_price_below_both_ma | is_price_below_slow_ma & is_slow_ma_trend_bearish | is_price_below_fast_ma & is_fast_ma_trend_bearish | is_price_below_both_ma & is_both_ma_trend_bearish | is_slow_ma_trend_bearish | is_fast_ma_trend_bearish)

    # Result conditions
    result_green_elephant = final_green_elephant & activate_green_elephant & ((filter_type == 'CON FILTRADO DE TENDENCIA') | ((filter_type == 'SIN FILTRADO DE TENDENCIA') & is_green_strong))
    result_red_elephant = final_red_elephant & activate_red_elephant & ((filter_type == 'CON FILTRADO DE TENDENCIA') | ((filter_type == 'SIN FILTRADO DE TENDENCIA') & is_red_strong))

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(fast_ma.iloc[i]) or pd.isna(slow_ma.iloc[i]) or pd.isna(atr.iloc[i]):
            continue

        if result_green_elephant.iloc[i] and not is_excluded_time.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

        if result_red_elephant.iloc[i] and not is_excluded_time.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
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