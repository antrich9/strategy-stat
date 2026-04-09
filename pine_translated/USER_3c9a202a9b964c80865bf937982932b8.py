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

    length = 10
    minMult = 1.0
    maxMult = 5.0
    step = 0.5
    perfAlpha = 10.0
    fromCluster = 'Best'

    close = df['close']
    high = df['high']
    low = df['low']
    hl2 = (high + low) / 2.0

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/length, adjust=False).mean()

    factors = np.arange(minMult, maxMult + step/2, step).tolist()
    n_factors = len(factors)
    holder = np.full(n_factors, hl2.iloc[0])

    entries = []
    trade_num = 1
    prev_os = None

    for i in range(len(df)):
        if pd.isna(close.iloc[i]) or pd.isna(hl2.iloc[i]) or pd.isna(atr.iloc[i]):
            prev_os = None
            continue

        if i == 0:
            holder = np.full(n_factors, hl2.iloc[i])
            prev_os = 1
            continue

        current_close = close.iloc[i]
        current_hl2 = hl2.iloc[i]
        current_atr = atr.iloc[i]

        for k in range(n_factors):
            up = current_hl2 + current_atr * factors[k]
            dn = current_hl2 - current_atr * factors[k]
            if current_close > holder[k]:
                holder[k] = min(up, holder[k])
            else:
                holder[k] = max(dn, holder[k])

        data_arr = [current_close - holder[k] for k in range(n_factors)]
        factor_arr = [holder[k] for k in range(n_factors)]

        p25 = np.percentile(data_arr, 25)
        p50 = np.percentile(data_arr, 50)
        p75 = np.percentile(data_arr, 75)

        centroids = [p25, p50, p75]

        new_centroids = centroids

        from_idx = 2 if fromCluster == 'Best' else (1 if fromCluster == 'Average' else 0)

        avg_data = np.mean(data_arr)
        abs_diff = abs(current_close - close.iloc[i-1]) if i > 0 else 0

        alpha_ema = 1.0 / int(perfAlpha)
        if i == 1:
            ema_abs_diff = abs_diff
        else:
            ema_abs_diff = alpha_ema * abs_diff + (1 - alpha_ema) * ema_abs_diff

        perf_idx = max(avg_data, 0) / ema_abs_diff if ema_abs_diff != 0 else 0
        signal_rating = perf_idx * 10

        target_factor = np.mean(factor_arr)
        up_main = current_hl2 + current_atr * target_factor
        dn_main = current_hl2 - current_atr * target_factor

        if current_close > up_main:
            os = 1
        elif current_close < dn_main:
            os = 0
        else:
            os = prev_os if prev_os is not None else 1

        long_cond = signal_rating >= 7 and os == 1
        short_cond = signal_rating >= 7 and os == 0

        if long_cond:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1

        if short_cond:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1

        prev_os = os

    return entries