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
    high = df['high']
    low = df['low']
    n = len(df)

    tau = 1
    evolution_steps = 10
    nn_lookback = 40
    signal_smoothing = 3
    chaos_threshold = 0.05
    stability_threshold = -0.05
    trade_chaos = True
    trade_stable = True
    trend_filter = True
    sma_len = 20

    ret = close.pct_change()
    roll_std = ret.rolling(20).std()
    norm_ret = ret / roll_std
    norm_ret = norm_ret.replace([np.inf, -np.inf], 0)
    norm_ret = norm_ret.fillna(0)

    def find_nn_idx(i):
        if i < evolution_steps + 1 or i >= n:
            return evolution_steps + 1
        best_dist = 1e10
        best_j = evolution_steps + 1
        norm_ret_0 = norm_ret.iloc[i]
        for j in range(evolution_steps + 1, min(nn_lookback + 1, i + 1)):
            d = abs(norm_ret.iloc[i - j] - norm_ret_0)
            if d < best_dist:
                best_dist = d
                best_j = j
        return best_j

    lyapunov_raw = pd.Series(0.0, index=df.index)
    for i in range(evolution_steps + nn_lookback, n):
        nn_offset = find_nn_idx(i)
        div_sum = 0.0
        cnt = 0
        norm_ret_0 = norm_ret.iloc[i]
        for step in range(1, evolution_steps + 1):
            cur_shifted = norm_ret.iloc[i + step] if i + step < n else norm_ret.iloc[n - 1]
            nbr_shifted = norm_ret.iloc[i - nn_offset + step] if i - nn_offset + step < n else norm_ret.iloc[n - 1]
            d_now = abs(norm_ret_0 - norm_ret.iloc[i - nn_offset])
            d_future = abs(cur_shifted - nbr_shifted)
            if d_now > 1e-10 and d_future > 1e-10:
                div_sum += np.log(d_future / d_now) / step
                cnt += 1
        lyapunov_raw.iloc[i] = div_sum / cnt if cnt > 0 else 0.0

    lyapunov = lyapunov_raw.ewm(span=signal_smoothing, adjust=False).mean()

    is_chaotic = lyapunov > chaos_threshold
    is_stable = lyapunov < stability_threshold

    chaos_entry = pd.Series(False, index=df.index)
    stability_entry = pd.Series(False, index=df.index)
    for i in range(1, n):
        if not np.isnan(lyapunov.iloc[i]) and not np.isnan(lyapunov.iloc[i-1]):
            if lyapunov.iloc[i] > chaos_threshold and lyapunov.iloc[i-1] <= chaos_threshold:
                chaos_entry.iloc[i] = True
            if lyapunov.iloc[i] < stability_threshold and lyapunov.iloc[i-1] >= stability_threshold:
                stability_entry.iloc[i] = True

    sma20 = close.rolling(sma_len).mean()
    trend_up = close > sma20
    trend_down = close < sma20

    price_high20 = high.rolling(20).max()
    price_low20 = low.rolling(20).min()
    pct_rank = (close - price_low20) / price_high20.clip(lower=1e-10) - price_low20

    chaos_long = trade_chaos & chaos_entry & ((not trend_filter) | trend_up)
    chaos_short = trade_chaos & chaos_entry & ((not trend_filter) | trend_down)
    stable_long = trade_stable & stability_entry & (pct_rank < 0.25)
    stable_short = trade_stable & stability_entry & (pct_rank > 0.75)

    go_long = chaos_long | stable_long
    go_short = chaos_short | stable_short

    entries = []
    trade_num = 1
    in_position = False
    position_direction = None

    for i in range(n):
        if np.isnan(lyapunov.iloc[i]):
            continue
        if go_long.iloc[i] and not in_position:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
            in_position = True
            position_direction = 'long'
        elif go_short.iloc[i] and not in_position:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
            in_position = True
            position_direction = 'short'

    return entries