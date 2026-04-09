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
    # Input parameters
    fastLength = 13
    slowLength = 3
    almaOffset = 0.85
    almaSigma = 6.0
    midTrendPeriod = 34
    smallTrendPeriod = 5
    tinyTrendPeriod = 3
    overboughtThreshold = 80
    oversoldThreshold = 20

    # Helper: find_recent_value
    def find_recent_value(values, length):
        recent_value = np.nan
        for i in range(length + 1):
            if i < len(values):
                if np.isnan(recent_value) or not np.isnan(values.iloc[i]):
                    recent_value = values.iloc[i]
        return recent_value

    # Helper: weighted simple average
    def calculate_weighted_simple_average(src, length, weight):
        result = pd.Series(index=src.index, dtype=float)
        sum_float = np.nan
        output = np.nan
        for i in range(len(src)):
            if i >= length:
                val = src.iloc[i - length]
            else:
                val = src.iloc[i]
            if pd.notna(val):
                if pd.isna(sum_float):
                    sum_float = 0.0
                    for j in range(length):
                        if i - length + j >= 0 and pd.notna(src.iloc[i - length + j]):
                            sum_float += src.iloc[i - length + j]
                else:
                    if i - length >= 0 and pd.notna(src.iloc[i - length]):
                        sum_float -= src.iloc[i - length]
                    if pd.notna(src.iloc[i]):
                        sum_float += src.iloc[i]
            if pd.notna(sum_float) and sum_float != 0:
                moving_average = sum_float / length
            else:
                moving_average = np.nan
            if pd.isna(output):
                output = moving_average
            else:
                if pd.notna(moving_average) and pd.notna(output):
                    output = (src.iloc[i] * weight + output * (length - weight)) / length
                else:
                    output = moving_average
            result.iloc[i] = output
        return result

    # Helper: ALMA
    def alma(src, length, offset, sigma):
        result = pd.Series(index=src.index, dtype=float)
        m = offset * (length - 1)
        s = sigma * length / 6.0
        w = pd.Series(1.0, index=src.index)
        for i in range(length):
            w.iloc[len(w) - length + i] = np.exp(-((i - m) ** 2) / (2 * s * s))
        w_sum = w.iloc[-length:].sum()
        for i in range(len(src)):
            if i < length - 1:
                result.iloc[i] = np.nan
            else:
                start_idx = i - length + 1
                end_idx = i + 1
                window = src.iloc[start_idx:end_idx]
                weights = w.iloc[-length:]
                valid = window.notna() & weights.notna()
                if valid.sum() > 0:
                    result.iloc[i] = (window[valid] * weights[valid]).sum() / weights[valid].sum()
                else:
                    result.iloc[i] = np.nan
        return result

    # ESCGO Calculation
    src = (df['high'] + df['low'] + 2 * df['close'] + df['open'] / 2) / 4.5

    cg = pd.Series(index=df.index, dtype=float)
    for i in range(fastLength - 1, len(df)):
        num = 0.0
        denom = 0.0
        for j in range(fastLength):
            idx = i - fastLength + 1 + j
            if idx >= 0 and idx < len(src):
                val = src.iloc[idx]
                if pd.notna(val):
                    num += (1 + j) * val
                    denom += val
        cg.iloc[i] = (-num / denom) + ((fastLength + 1) / 2) if denom != 0 else 0

    maxc = cg.rolling(fastLength).max()
    minc = cg.rolling(fastLength).min()
    v1 = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if maxc.iloc[i] != minc.iloc[i]:
            v1.iloc[i] = (cg.iloc[i] - minc.iloc[i]) / (maxc.iloc[i] - minc.iloc[i])
        else:
            v1.iloc[i] = 0.0

    v2_ = (4 * v1 + 3 * v1.shift(1) + 2 * v1.shift(2) + v1.shift(3)) / 10.0
    escgo = 2 * (v2_ - 0.5)
    escgolag = alma(escgo, slowLength, almaOffset, almaSigma)

    escgoBullish = escgo > escgolag
    escgoBearish = escgo < escgolag
    escgoBullishCross = (escgo > escgolag) & (escgo.shift(1) <= escgolag.shift(1))
    escgoBearishCross = (escgo < escgolag) & (escgo.shift(1) >= escgolag.shift(1))

    # Banker Fund Flow
    typical_price = (2 * df['close'] + df['high'] + df['low'] + df['open']) / 5
    lowest_low = df['low'].rolling(midTrendPeriod).min()
    highest_high = df['high'].rolling(midTrendPeriod).max()

    small_sum_period = smallTrendPeriod + tinyTrendPeriod - 1
    inner_val = (df['close'] - df['low'].rolling(small_sum_period).min()) / (df['high'].rolling(small_sum_period).max() - df['low'].rolling(small_sum_period).min()) * 100
    inner_val = inner_val.fillna(0)

    fund_flow_smooth1 = calculate_weighted_simple_average(inner_val, smallTrendPeriod, 1)
    fund_flow_smooth2 = calculate_weighted_simple_average(fund_flow_smooth1, tinyTrendPeriod, 1)
    fund_flow_trend = (3 * fund_flow_smooth1 - 2 * fund_flow_smooth2 - 50) * 1.032 + 50

    bull_bear_line = ((typical_price - lowest_low) / (highest_high - lowest_low) * 100).ewm(span=13, adjust=False).mean()

    banker_entry_signal = crossover(fund_flow_trend, bull_bear_line) & (bull_bear_line < 25)
    bankerYellow = banker_entry_signal
    bankerGreen = fund_flow_trend > bull_bear_line
    bankerWhite = fund_flow_trend < find_recent_value(fund_flow_trend * 0.95, 1)
    bankerRed = fund_flow_trend < bull_bear_line

    # Signal Fusion
    longEntry = escgoBullishCross & (bankerYellow | bankerGreen)
    shortEntry = escgoBearishCross & (bankerRed | bankerWhite)

    # Generate entries
    entries = []
    trade_num = 1
    in_position = False

    for i in range(1, len(df)):
        if pd.isna(escgo.iloc[i]) or pd.isna(escgolag.iloc[i]):
            continue
        if pd.isna(fund_flow_trend.iloc[i]) or pd.isna(bull_bear_line.iloc[i]):
            continue

        if not in_position:
            if longEntry.iloc[i]:
                ts = int(df['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                })
                trade_num += 1
                in_position = True
            elif shortEntry.iloc[i]:
                ts = int(df['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(df['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(df['close'].iloc[i]),
                    'raw_price_b': float(df['close'].iloc[i])
                })
                trade_num += 1
                in_position = True
        else:
            in_position = False

    return entries