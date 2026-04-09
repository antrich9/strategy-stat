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
    n = len(df)
    results = []
    trade_num = 1

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_arr = df['open'].values
    volume = df['volume'].values
    timestamp = df['time'].values

    # ==========================================
    # SUPERTREND CALCULATION
    # ==========================================
    Periods_ST = 10
    src_ST = (high + low) / 2  # hl2
    Multiplier_ST = 3.0

    # Wilder ATR
    tr_st = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_st[0] = high[0] - low[0]
    atr_st = np.zeros(n)
    atr_st[0] = tr_st[0]
    for i in range(1, n):
        atr_st[i] = (atr_st[i-1] * (Periods_ST - 1) + tr_st[i]) / Periods_ST

    up = np.zeros(n)
    up1 = np.zeros(n)
    dn = np.zeros(n)
    dn1 = np.zeros(n)
    trend = np.zeros(n)

    up[0] = src_ST[0] - Multiplier_ST * atr_st[0]
    up1[0] = up[0]
    dn[0] = src_ST[0] + Multiplier_ST * atr_st[0]
    dn1[0] = dn[0]
    trend[0] = 1

    for i in range(1, n):
        up[i] = src_ST[i] - Multiplier_ST * atr_st[i]
        dn[i] = src_ST[i] + Multiplier_ST * atr_st[i]
        if close[i-1] > up1[i-1]:
            up[i] = max(up[i], up1[i-1])
        if close[i-1] < dn1[i-1]:
            dn[i] = min(dn[i], dn1[i-1])
        up1[i] = up[i]
        dn1[i] = dn[i]
        if trend[i-1] == -1 and close[i] > dn1[i-1]:
            trend[i] = 1
        elif trend[i-1] == 1 and close[i] < up1[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

    buySignal = (trend == 1) & (np.roll(trend, 1) == -1)
    buySignal[0] = False

    # ==========================================
    # ZERO LAG MACD (MLB) CALCULATION
    # ==========================================
    fast_lengthMLB = 14
    slow_lengthMLB = 28
    srcMLB = close.copy()
    signalMLB_lengthMLB = 11
    gammaMLB = 0.02
    use_lagMLBMLB = True

    def get_zlema(src, length):
        lag = int((length - 1) / 2)
        adjusted = src + src - np.roll(src, lag)
        adjusted[0] = src[0]
        return pd.Series(adjusted).ewm(span=length, adjust=False).mean().values

    def laguerre(g, p):
        L0 = np.zeros(n)
        L1 = np.zeros(n)
        L2 = np.zeros(n)
        L3 = np.zeros(n)
        for i in range(1, n):
            L0[i] = (1 - g) * p[i] + g * L0[i-1]
            L1[i] = -g * L0[i] + L0[i-1] + g * L1[i-1]
            L2[i] = -g * L1[i] + L1[i-1] + g * L2[i-1]
            L3[i] = -g * L2[i] + L2[i-1] + g * L3[i-1]
        return (L0 + 2 * L1 + 2 * L2 + L3) / 6

    def get_ma(src, length, ma_type):
        if ma_type == 'SMA':
            return pd.Series(src).rolling(length).mean().values
        elif ma_type == 'EMA':
            return pd.Series(src).ewm(span=length, adjust=False).mean().values
        elif ma_type == 'ZLEMA':
            return get_zlema(src, length)
        return src

    source_typeMLB = 'ZLEMA'
    signalMLB_typeMLB = 'ZLEMA'

    fast_maMLB = get_ma(srcMLB, fast_lengthMLB, source_typeMLB)
    slow_maMLB = get_ma(srcMLB, slow_lengthMLB, source_typeMLB)
    macdMLB = fast_maMLB - slow_maMLB

    fast_leader = get_ma(srcMLB - fast_maMLB, fast_lengthMLB, source_typeMLB)
    slow_leader = get_ma(srcMLB - slow_maMLB, slow_lengthMLB, source_typeMLB)
    macdMLB_leader = fast_maMLB + fast_leader - (slow_maMLB + slow_leader)

    if use_lagMLBMLB:
        macdMLB = laguerre(gammaMLB, macdMLB)
        macdMLB_leader = laguerre(gammaMLB, macdMLB_leader)

    signalMLB = get_ma(macdMLB, signalMLB_lengthMLB, signalMLB_typeMLB)
    histMLB = macdMLB - signalMLB

    # ==========================================
    # VOLUME FLOW CONDITION
    # ==========================================
    greenLine = pd.Series(close).rolling(14).mean().values
    orangeLine = pd.Series(close).rolling(28).mean().values
    volumeFlowCondition = greenLine > orangeLine

    # ==========================================
    # ENTRY CONDITIONS
    # ==========================================
    bullishEntryCondition = (histMLB > 0) & volumeFlowCondition & buySignal

    # ==========================================
    # GENERATE ENTRIES
    # ==========================================
    for i in range(n):
        if bullishEntryCondition[i]:
            entry_price = close[i]
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(timestamp[i]),
                'entry_time': datetime.fromtimestamp(timestamp[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return results