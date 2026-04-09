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
    tradeDirection = "Both"
    lengthVidya = 14
    lengthCmo = 14
    alpha = 0.2
    cmoLength = 14
    fiLength = 13
    emaLength = 13
    input_lookback = 20
    input_retSince = 2
    input_retValid = 2
    rTon = True
    rTcc = False
    rThv = False

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    time = df['time'].values

    n = len(df)

    # CMO calculation (Wilder RSI based)
    def calc_cmo(src, length):
        mom = np.diff(src, prepend=src[0])
        gain = np.where(mom > 0, mom, 0)
        loss = np.where(mom < 0, -mom, 0)
        avg_gain = np.zeros(len(src))
        avg_loss = np.zeros(len(src))
        if len(src) > 0:
            avg_gain[0] = gain[0]
            avg_loss[0] = loss[0]
        for i in range(1, len(src)):
            avg_gain[i] = (avg_gain[i-1] * (length - 1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length - 1) + loss[i]) / length
        rsi = np.where(avg_loss == 0, 100, 100 - (100 / (1 + avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss))))
        cmo_val = np.where(rsi >= 50, rsi - 50, rsi - 50)
        return cmo_val

    cmo = calc_cmo(close, cmoLength)

    # VIDYA calculation
    vidya = np.zeros(n)
    for i in range(n):
        cmo_val = cmo[i]
        if i == 0:
            vidya[i] = 0
        else:
            vidya[i] = vidya[i-1] + alpha * (cmo_val / 100) * (close[i] - vidya[i-1])

    # Force Index
    force_index_raw = np.zeros(n)
    for i in range(1, n):
        force_index_raw[i] = (close[i] - close[i-1]) * volume[i]
    force_index_ema = pd.Series(force_index_raw).ewm(span=fiLength, adjust=False).mean().values

    # Pivot points
    bb = input_lookback
    pl = np.full(n, np.nan)
    ph = np.full(n, np.nan)
    for i in range(bb, n):
        lowest_idx = np.argmin(low[i-bb:i+bb+1])
        if lowest_idx == bb:
            pl[i] = low[i]
        highest_idx = np.argmax(high[i-bb:i+bb+1])
        if highest_idx == bb:
            ph[i] = high[i]

    # Box values
    sBox_top = np.full(n, np.nan)
    sBox_bot = np.full(n, np.nan)
    rBox_top = np.full(n, np.nan)
    rBox_bot = np.full(n, np.nan)

    for i in range(n):
        if not np.isnan(pl[i]):
            s_yLoc = low[i - bb + 1] if low[i - bb + 1] > low[i - bb - 1] else low[i - bb - 1]
            sBox_bot[i] = pl[i]
            sBox_top[i] = s_yLoc

        if not np.isnan(ph[i]):
            r_yLoc = high[i - bb + 1] if high[i - bb + 1] > high[i - bb - 1] else high[i - bb - 1]
            rBox_top[i] = ph[i]
            rBox_bot[i] = r_yLoc

    # Breakout flags
    sBreak = np.zeros(n, dtype=bool)
    rBreak = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if rThv:
            cu = close[i] < sBox_bot[i-1] and close[i-1] >= sBox_bot[i-1]
            co = high[i] > rBox_top[i-1] and high[i-1] <= rBox_top[i-1]
        elif rTcc:
            cu = close[i] < sBox_bot[i-1] and close[i-1] >= sBox_bot[i-1]
            co = close[i] > rBox_top[i-1] and close[i-1] <= rBox_top[i-1]
        else:
            cu = close[i] < sBox_bot[i-1] and close[i-1] >= sBox_bot[i-1]
            co = close[i] > rBox_top[i-1] and close[i-1] <= rBox_top[i-1]

        if cu and not sBreak[i-1]:
            sBreak[i] = True
        else:
            sBreak[i] = sBreak[i-1]

        if co and not rBreak[i-1]:
            rBreak[i] = True
        else:
            rBreak[i] = rBreak[i-1]

        if not np.isnan(pl[i]) and not sBreak[i]:
            sBreak[i] = False
        if not np.isnan(ph[i]) and not rBreak[i]:
            rBreak[i] = False

    # Retest conditions
    s1 = np.zeros(n, dtype=bool)
    s2 = np.zeros(n, dtype=bool)
    s3 = np.zeros(n, dtype=bool)
    s4 = np.zeros(n, dtype=bool)
    r1 = np.zeros(n, dtype=bool)
    r2 = np.zeros(n, dtype=bool)
    r3 = np.zeros(n, dtype=bool)
    r4 = np.zeros(n, dtype=bool)

    for i in range(bb + input_retSince + 1, n):
        bars_since_sBreak = 0
        for j in range(i-1, -1, -1):
            if sBreak[j]:
                break
            bars_since_sBreak += 1

        bars_since_rBreak = 0
        for j in range(i-1, -1, -1):
            if rBreak[j]:
                break
            bars_since_rBreak += 1

        if bars_since_sBreak > input_retSince:
            if not np.isnan(sBox_top[i-1]) and not np.isnan(sBox_bot[i-1]):
                s1[i] = high[i] >= sBox_top[i-1] and close[i] <= sBox_bot[i-1]
                s2[i] = high[i] >= sBox_top[i-1] and close[i] >= sBox_bot[i-1] and close[i] <= sBox_top[i-1]
                s3[i] = high[i] >= sBox_bot[i-1] and high[i] <= sBox_top[i-1]
                s4[i] = high[i] >= sBox_bot[i-1] and high[i] <= sBox_top[i-1] and close[i] < sBox_bot[i-1]

        if bars_since_rBreak > input_retSince:
            if not np.isnan(rBox_top[i-1]) and not np.isnan(rBox_bot[i-1]):
                r1[i] = low[i] <= rBox_bot[i-1] and close[i] >= rBox_top[i-1]
                r2[i] = low[i] <= rBox_bot[i-1] and close[i] <= rBox_top[i-1] and close[i] >= rBox_bot[i-1]
                r3[i] = low[i] <= rBox_top[i-1] and low[i] >= rBox_bot[i-1]
                r4[i] = low[i] <= rBox_top[i-1] and low[i] >= rBox_bot[i-1] and close[i] > rBox_top[i-1]

    # Retest valid signal
    retValid_long = np.zeros(n, dtype=bool)
    retValid_short = np.zeros(n, dtype=bool)

    for i in range(bb + input_retSince + input_retValid + 1, n):
        support_ret = s1[i] or s2[i] or s3[i] or s4[i]
        resistance_ret = r1[i] or r2[i] or r3[i] or r4[i]

        bars_since_support_ret = 0
        for j in range(i-1, -1, -1):
            if s1[j] or s2[j] or s3[j] or s4[j]:
                break
            bars_since_support_ret += 1

        bars_since_resistance_ret = 0
        for j in range(i-1, -1, -1):
            if r1[j] or r2[j] or r3[j] or r4[j]:
                break
            bars_since_resistance_ret += 1

        if bars_since_support_ret > 0 and bars_since_support_ret <= input_retValid:
            retValid_long[i] = support_ret

        if bars_since_resistance_ret > 0 and bars_since_resistance_ret <= input_retValid:
            retValid_short[i] = resistance_ret

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(n):
        if retValid_long[i] and (tradeDirection == "Long" or tradeDirection == "Both"):
            entry_ts = int(time[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close[i]),
                'raw_price_b': float(close[i])
            })
            trade_num += 1

        if retValid_short[i] and (tradeDirection == "Short" or tradeDirection == "Both"):
            entry_ts = int(time[i])
            entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(close[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close[i]),
                'raw_price_b': float(close[i])
            })
            trade_num += 1

    return entries