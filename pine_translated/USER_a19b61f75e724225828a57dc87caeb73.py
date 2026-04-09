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
    min_body_pct = 70
    lookback_bars = 100
    atr_factor = 1.3
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0

    high = df['high']
    low = df['low']
    close = df['close']
    open_price = df['open']

    prev_tr = pd.Series([high.iloc[i] - low.iloc[i]] + [max(high.iloc[i] - low.iloc[i], abs(high.iloc[i] - close.iloc[i-1]), abs(low.iloc[i] - close.iloc[i-1])) for i in range(1, len(df))], index=df.index)
    atr_val = prev_tr.ewm(alpha=1.0/lookback_bars, adjust=False).mean()

    is_green = close > open_price
    is_red = close < open_price
    body_size = (close - open_price).abs()
    total_range = high - low
    body_pct = np.where(total_range > 0, (body_size / total_range) * 100, 0)

    atr_prev = atr_val.shift(1)
    elephant_green = is_green & (body_pct >= min_body_pct) & (body_size >= atr_prev * atr_factor)
    elephant_red = is_red & (body_pct >= min_body_pct) & (body_size >= atr_prev * atr_factor)

    pct_change = close.shift(trendilo_smooth) / close * 100
    pct_change.iloc[0] = 0

    def alma_indicator(data, window, offset, sigma):
        result = pd.Series(index=data.index, dtype=float)
        m = (window - 1) * offset
        s = sigma * (window - 1) / 6.0
        if s == 0:
            result = data.rolling(window).mean()
        else:
            for i in range(window - 1, len(data)):
                total_weight = 0.0
                weighted_sum = 0.0
                for k in range(window):
                    weight = np.exp(-((k - m) ** 2) / (2 * s * s))
                    weighted_sum += data.iloc[i - window + 1 + k] * weight
                    total_weight += weight
                result.iloc[i] = weighted_sum / total_weight if total_weight > 0 else data.iloc[i]
        return result

    avg_pct_change = alma_indicator(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)

    rms = trendilo_bmult * np.sqrt(avg_pct_change.rolling(trendilo_length).apply(lambda x: (x * x).sum() / trendilo_length, raw=True))

    trendilo_dir = pd.Series(index=df.index, dtype=int)
    trendilo_dir[avg_pct_change > rms] = 1
    trendilo_dir[avg_pct_change < -rms] = -1
    trendilo_dir[(avg_pct_change <= rms) & (avg_pct_change >= -rms)] = 0

    trendilo_bull = trendilo_dir == 1
    trendilo_bear = trendilo_dir == -1

    long_condition = elephant_green & trendilo_bull
    short_condition = elephant_red & trendilo_bear

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i < lookback_bars:
            continue
        if pd.isna(trendilo_dir.iloc[i]) or pd.isna(atr_val.iloc[i]) or pd.isna(elephant_green.iloc[i]):
            continue

        direction = None
        if long_condition.iloc[i]:
            direction = 'long'
        elif short_condition.iloc[i]:
            direction = 'short'

        if direction is None:
            continue

        ts = int(df['time'].iloc[i])
        entries.append({
            'trade_num': trade_num,
            'direction': direction,
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