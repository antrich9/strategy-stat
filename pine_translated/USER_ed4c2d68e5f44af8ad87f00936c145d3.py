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
    entries = []
    trade_num = 1

    # Settings
    length = 10
    minMult = 1
    maxMult = 5
    step = 0.5
    perfAlpha = 10
    fromCluster = 'Best'
    maxIter = 1000
    maxData = 10000

    from_idx = {'Best': 2, 'Average': 1, 'Worst': 0}[fromCluster]

    # Calculate hl2
    hl2 = (df['high'] + df['low']) / 2.0

    # Calculate Wilder ATR
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(index=df.index, dtype=float)
    atr.iloc[length - 1] = tr.iloc[:length].mean()
    multiplier = 1.0 - (1.0 / length)
    for i in range(length, len(df)):
        atr.iloc[i] = tr.iloc[i] * (1.0 / length) + atr.iloc[i - 1] * multiplier

    # Generate factors array
    factors = []
    i = 0
    current = minMult
    while current <= maxMult + 0.001:
        factors.append(current)
        i += 1
        current = minMult + i * step

    n_factors = len(factors)

    # Initialize holder arrays
    upper_arr = pd.DataFrame(index=df.index, columns=range(n_factors), dtype=float)
    lower_arr = pd.DataFrame(index=df.index, columns=range(n_factors), dtype=float)
    trend_arr = pd.DataFrame(index=df.index, columns=range(n_factors), dtype=int)
    output_arr = pd.DataFrame(index=df.index, columns=range(n_factors), dtype=float)
    perf_arr = pd.DataFrame(index=df.index, columns=range(n_factors), dtype=float)
    factor_arr = pd.DataFrame(index=df.index, columns=range(n_factors), dtype=float)

    # Initialize first values
    upper_arr.iloc[0] = hl2.iloc[0]
    lower_arr.iloc[0] = hl2.iloc[0]
    trend_arr.iloc[0] = 0
    output_arr.iloc[0] = hl2.iloc[0]
    perf_arr.iloc[0] = 0
    factor_arr.iloc[0] = factors

    # Calculate SuperTrend for all factors
    for k in range(n_factors):
        factor = factors[k]
        for i in range(1, len(df)):
            up = hl2.iloc[i] + atr.iloc[i] * factor
            dn = hl2.iloc[i] - atr.iloc[i] * factor

            prev_upper = upper_arr.iloc[i - 1, k]
            prev_lower = lower_arr.iloc[i - 1, k]
            prev_trend = trend_arr.iloc[i - 1, k]
            prev_output = output_arr.iloc[i - 1, k]
            prev_perf = perf_arr.iloc[i - 1, k]

            close_curr = df['close'].iloc[i]
            close_prev = df['close'].iloc[i - 1]

            new_upper = up if close_prev >= prev_upper else min(up, prev_upper)
            new_lower = dn if close_prev <= prev_lower else max(dn, prev_lower)

            if close_curr > new_upper:
                new_trend = 1
            elif close_curr < new_lower:
                new_trend = 0
            else:
                new_trend = prev_trend

            diff = 0 if pd.isna(close_prev - prev_output) else np.sign(close_prev - prev_output)
            if pd.isna(prev_perf):
                new_perf = 0
            else:
                new_perf = prev_perf + (2.0 / (perfAlpha + 1)) * ((close_curr - close_prev) * diff - prev_perf)

            new_output = new_lower if new_trend == 1 else new_upper

            upper_arr.iloc[i, k] = new_upper
            lower_arr.iloc[i, k] = new_lower
            trend_arr.iloc[i, k] = new_trend
            output_arr.iloc[i, k] = new_output
            perf_arr.iloc[i, k] = new_perf
            factor_arr.iloc[i, k] = factor

    # K-means clustering
    last_bar_index = len(df) - 1
    max_data_idx = max(0, last_bar_index - maxData)

    if max_data_idx < len(df):
        perf_data = perf_arr.iloc[max_data_idx:].values.flatten()
        perf_data = perf_data[~np.isnan(perf_data)]

        factor_data = factor_arr.iloc[max_data_idx:].values.flatten()
        factor_data = factor_data[~np.isnan(factor_data)]

        if len(perf_data) > 0 and len(factor_data) >= 3:
            sorted_idx = np.argsort(perf_data)
            c1_idx = int(0.25 * (len(sorted_idx) - 1))
            c2_idx = int(0.50 * (len(sorted_idx) - 1))
            c3_idx = int(0.75 * (len(sorted_idx) - 1))

            centroids = np.array([
                perf_data[sorted_idx[c1_idx]],
                perf_data[sorted_idx[c2_idx]],
                perf_data[sorted_idx[c3_idx]]
            ])

            factors_clusters = [np.array([]) for _ in range(3)]
            perfclusters = [np.array([]) for _ in range(3)]

            for iteration in range(maxIter):
                clusters_factors = [np.array([]) for _ in range(3)]
                clusters_perf = [np.array([]) for _ in range(3)]

                for i, value in enumerate(perf_data):
                    if i < len(factor_data):
                        distances = np.abs(centroids - value)
                        idx = np.argmin(distances)
                        clusters_factors[idx] = np.append(clusters_factors[idx], factor_data[i])
                        clusters_perf[idx] = np.append(clusters_perf[idx], value)

                new_centroids = np.array([c.mean() if len(c) > 0 else centroids[i] for i, c in enumerate(clusters_perf)])

                if (np.isclose(new_centroids[0], centroids[0]) and
                    np.isclose(new_centroids[1], centroids[1]) and
                    np.isclose(new_centroids[2], centroids[2])):
                    break

                centroids = new_centroids
                factors_clusters = clusters_factors
                perfclusters = clusters_perf

            target_factor = factors_clusters[from_idx].mean() if len(factors_clusters[from_idx]) > 0 else factors[4]
        else:
            target_factor = factors[4]
    else:
        target_factor = factors[4]

    # Calculate final SuperTrend and os
    den = (df['close'].diff().abs()).ewm(span=int(perfAlpha), adjust=False).mean()

    upper = hl2.iloc[0]
    lower = hl2.iloc[0]
    os = 0
    perf_ama = np.nan
    perf_idx = 0

    os_series = pd.Series(index=df.index, dtype=int)
    perf_idx_series = pd.Series(index=df.index, dtype=float)

    for i in range(len(df)):
        up = hl2.iloc[i] + atr.iloc[i] * target_factor
        dn = hl2.iloc[i] - atr.iloc[i] * target_factor

        if i > 0:
            prev_upper = upper
            prev_lower = lower
        else:
            prev_upper = upper
            prev_lower = lower

        upper = up if df['close'].iloc[i - 1] < prev_upper else min(up, prev_upper) if i > 0 else up
        lower = dn if df['close'].iloc[i - 1] > prev_lower else max(dn, prev_lower) if i > 0 else dn

        if df['close'].iloc[i] > upper:
            os = 1
        elif df['close'].iloc[i] < lower:
            os = 0

        os_series.iloc[i] = os

        if den.iloc[i] > 0:
            perf_idx = max(perfclusters[from_idx].mean() if len(perfclusters) > 0 and len(perfclusters[from_idx]) > 0 else 0, 0) / den.iloc[i]
        else:
            perf_idx = 0

        perf_idx_series.iloc[i] = perf_idx

    # Calculate signal_rating
    signal_rating = perf_idx_series * 10

    # Entry conditions
    longCondition = (signal_rating >= 7) & (os_series == 1)
    shortCondition = (signal_rating >= 7) & (os_series == 0)

    # Detect crossovers for entries
    prev_os = os_series.shift(1).fillna(0).astype(int)

    long_entry = longCondition & (prev_os < os_series)
    short_entry = shortCondition & (prev_os > os_series)

    # Generate entries
    for i in range(1, len(df)):
        if long_entry.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
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
        elif short_entry.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
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