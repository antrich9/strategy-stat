import pandas as pd
import numpy as np
from datetime import datetime, timezone

def wilder_rsi(src, length):
    delta = src.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    n = len(src)
    rsi = np.full(n, np.nan)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    sum_gain = gains.iloc[:length].sum()
    sum_loss = losses.iloc[:length].sum()
    avg_gain[length - 1] = sum_gain / length
    avg_loss[length - 1] = sum_loss / length
    for i in range(length, n):
        avg_gain[i] = (avg_gain[i - 1] * (length - 1) + gains.iloc[i]) / length
        avg_loss[i] = (avg_loss[i - 1] * (length - 1) + losses.iloc[i]) / length
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=src.index)

def wilder_atr(high, low, close, length):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    n = len(tr)
    atr = np.full(n, np.nan)
    atr[length - 1] = tr.iloc[:length].mean()
    for i in range(length, n):
        atr[i] = (atr[i - 1] * (length - 1) + tr.iloc[i]) / length
    return pd.Series(atr, index=tr.index)

def calculate_roc(src, length):
    return ((src - src.shift(length)) / src.shift(length)) * 100

def calculate_wma(src, length):
    weights = np.arange(1, length + 1)
    return src.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_jma(df, length=7, phase=50, power=2):
    src = df['close'].copy()
    n = len(src)
    jma = np.zeros(n)
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = np.power(beta, power)
    phase_ratio = 0.5 if phase < -100 else (2.5 if phase > 100 else phase / 100 + 1.5)
    e0 = np.zeros(n)
    e1 = np.zeros(n)
    e2 = np.zeros(n)
    for i in range(1, n):
        e0[i] = (1 - alpha) * src.iloc[i] + alpha * e0[i - 1]
        e1[i] = (src.iloc[i] - e0[i]) * (1 - beta) + beta * e1[i - 1]
        e2[i] = (e0[i] + phase_ratio * e1[i] - jma[i - 1]) * np.power(1 - alpha, 2) + np.power(alpha, 2) * e2[i - 1]
        jma[i] = e2[i] + jma[i - 1]
    return pd.Series(jma, index=df.index)

def calculate_e2pss(df, period=15):
    price = (df['high'] + df['low']) / 2
    n = len(price)
    filt2 = np.zeros(n)
    trigger = np.zeros(n)
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * pi / period)
    b1 = 2 * a1 * np.cos(1.414 * pi / period)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    for i in range(n):
        if i < 2:
            filt2[i] = price.iloc[i]
        else:
            filt2[i] = coef1 * price.iloc[i] + coef2 * filt2[i - 1] + coef3 * filt2[i - 2]
        trigger[i] = filt2[i - 1] if i > 0 else 0
    return pd.Series(filt2, index=df.index), pd.Series(trigger, index=df.index)

def generate_entries(df: pd.DataFrame) -> list:
    src = df['close']
    high = df['high']
    low = df['low']
    use_CoppockCurve = True
    signalLogicCoppockCurve = 'Zero line'
    usejmaJMA = True
    usecolorjmaJMA = True
    inverseJMA = True
    useE2PSS = True
    inverseE2PSS = False
    PeriodE2PSS = 15
    roc_long = calculate_roc(src, 14)
    roc_short = calculate_roc(src, 11)
    coppock_raw = roc_long + roc_short
    coppock = calculate_wma(coppock_raw, 10)
    coppock_ma = coppock.ewm(span=10, adjust=False).mean()
    jmaJMA = calculate_jma(df)
    Filt2, TriggerE2PSS = calculate_e2pss(df, PeriodE2PSS)
    signalmaJMALong = (jmaJMA > jmaJMA.shift(1)) & (src > jmaJMA)
    signalmaJMAShort = (jmaJMA < jmaJMA.shift(1)) & (src < jmaJMA)
    finalLongSignalJMA = signalmaJMAShort
    finalShortSignalJMA = signalmaJMALong
    signalLongE2PSS = Filt2 > TriggerE2PSS
    signalShortE2PSS = Filt2 < TriggerE2PSS
    signalLongE2PSSFinal = signalShortE2PSS
    signalShortE2PSSFinal = signalLongE2PSS
    coppock_long_cond = coppock > 0
    coppock_short_cond = coppock < 0
    coppock_long = coppock_long_cond if signalLogicCoppockCurve == 'Zero line' else coppock > coppock_ma
    coppock_short = coppock_short_cond if signalLogicCoppockCurve == 'Zero line' else coppock < coppock_ma
    coppock_long = coppock_long if use_CoppockCurve else True
    coppock_short = coppock_short if use_CoppockCurve else True
    entrySignalLong = coppock_long & finalLongSignalJMA
    entrySignalShort = coppock_short & finalShortSignalJMA
    if useE2PSS:
        entrySignalLong = entrySignalLong & signalLongE2PSSFinal
        entrySignalShort = entrySignalShort & signalShortE2PSSFinal
    entries = []
    trade_num = 1
    for i in range(len(df)):
        entry_price = src.iloc[i]
        raw_price_a = entry_price
        raw_price_b = entry_price
        if entrySignalLong.iloc[i]:
            ts = df['time'].iloc[i]
            t_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': t_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': raw_price_a,
                'raw_price_b': raw_price_b
            })
            trade_num += 1
        elif entrySignalShort.iloc[i]:
            ts = df['time'].iloc[i]
            t_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': t_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': raw_price_a,
                'raw_price_b': raw_price_b
            })
            trade_num += 1
    return entries