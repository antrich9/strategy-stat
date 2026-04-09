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
    open_col = df['open']
    high_col = df['high']
    low_col = df['low']
    close_col = df['close']

    typicalPrice = (high_col + low_col + close_col) / 3

    cciPeriod6 = (typicalPrice - typicalPrice.rolling(6).mean()) / (0.015 * typicalPrice.rolling(5).std(ddof=1))
    cciPeriod10 = (typicalPrice - typicalPrice.rolling(10).mean()) / (0.015 * typicalPrice.rolling(9).std(ddof=1))
    cciPeriod20 = (typicalPrice - typicalPrice.rolling(20).mean()) / (0.015 * typicalPrice.rolling(19).std(ddof=1))
    cciPeriod30 = (typicalPrice - typicalPrice.rolling(30).mean()) / (0.015 * typicalPrice.rolling(29).std(ddof=1))
    cciPeriod60 = (typicalPrice - typicalPrice.rolling(60).mean()) / (0.015 * typicalPrice.rolling(59).std(ddof=1))

    weightedPriceAvg = (3 * close_col + high_col + low_col + open_col) / 6

    weights = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0])
    weightedMa = pd.Series(
        weightedPriceAvg.rolling(9).apply(lambda x: np.sum(x * weights) / 36 if len(x) == 9 else np.nan, raw=True),
        index=weightedPriceAvg.index
    )

    ll2 = low_col.rolling(2).min()
    ll4 = low_col.rolling(4).min()
    ll6 = low_col.rolling(6).min()
    midTermMa = (ll2 + ll4 + ll6) / 3

    shortTermLine = weightedMa
    midTermLine = midTermMa

    isUpTrend = shortTermLine > midTermLine
    isDownTrend = shortTermLine < midTermLine

    weightedTypicalPrice_cjdx = (2 * close_col + high_col + low_col) / 4
    tripleEmaValue = weightedTypicalPrice_cjdx.ewm(span=4, adjust=False).mean().ewm(span=4, adjust=False).mean().ewm(span=4, adjust=False).mean()
    jLineMomentum = (tripleEmaValue - tripleEmaValue.shift(1)) / tripleEmaValue.shift(1) * 100

    kLineFast = jLineMomentum.ewm(span=3, adjust=False).mean()
    dLineSlow = jLineMomentum.ewm(span=5, adjust=False).mean()

    isGoldenCross = (kLineFast > dLineSlow) & (kLineFast.shift(1) <= dLineSlow.shift(1))
    isDeathCross = (kLineFast < dLineSlow) & (kLineFast.shift(1) >= dLineSlow.shift(1))

    rawBuySignal = isGoldenCross & (jLineMomentum > 0)
    rawSellSignal = isDeathCross & (jLineMomentum < 0)

    signalGapPeriod = 3
    lastBuyBarIndex = [None]
    lastSellBarIndex = [None]

    filteredBuySignal = pd.Series(False, index=df.index)
    filteredSellSignal = pd.Series(False, index=df.index)

    for i in range(len(df)):
        if rawBuySignal.iloc[i]:
            if lastBuyBarIndex[0] is None or i - lastBuyBarIndex[0] >= signalGapPeriod:
                filteredBuySignal.iloc[i] = True
                lastBuyBarIndex[0] = i
        if rawSellSignal.iloc[i]:
            if lastSellBarIndex[0] is None or i - lastSellBarIndex[0] >= signalGapPeriod:
                filteredSellSignal.iloc[i] = True
                lastSellBarIndex[0] = i

    def xrf(values, length):
        if length < 1:
            return np.nan
        for i in range(length + 1):
            idx = length - i
            val = values.iloc[-idx] if idx <= len(values) else np.nan
            if not np.isnan(val):
                return val
        return np.nan

    def xsa(src, length, wei):
        result = src.ewm(span=length, adjust=False).mean()
        return result

    def xsl(src, length):
        lr = src.rolling(length).mean()
        lrprev = src.shift(1).rolling(length).mean()
        multiplier = 1
        return (lr - lrprev) / multiplier

    def xcn(cond, length):
        counts = cond.rolling(length).sum()
        return counts

    def xda(src, coeff):
        return src.ewm(alpha=coeff, adjust=False).mean()

    def xkdj(m, n1, n2):
        ed = (-0.4 * close_col.shift(4) - 0.4 * close_col.shift(3) - 1.1 * close_col.shift(2) + 0.9 * close_col.shift(1) + 2 * close_col)
        rsv = (ed - low_col.rolling(m).min()) / (high_col.rolling(m).max() - low_col.rolling(m).min()) * 100
        k = xsa(rsv, n1, 1)
        d = xsa(k, n2, 1)
        j = 2 * k - d
        return k, d, j

    x_1 = xsa((close_col - low_col.rolling(9).min()) / (high_col.rolling(9).max() - low_col.rolling(9).min()) * 100, 3, 1)
    x_2 = xsa((close_col - low_col.rolling(10).min()) / (high_col.rolling(10).max() - low_col.rolling(10).min()) * 100, 3, 1)
    x_3 = xsa((x_2 - 50) * 2, 3, 1) + xsa((x_1 - 50) * 2, 3, 1)

    godTrend = xsa((x_2 - 50) * 2, 3, 1) + xsa((x_1 - 50) * 2, 3, 1)

    xsl_close = xsl(close_col, 20) * 5 + close_col
    x_4 = (low_col > xsl_close.rolling(10).mean()) & (low_col < close_col.rolling(20).mean())

    sma5 = close_col.rolling(5).mean()
    sma10 = close_col.rolling(10).mean()
    crossunder_cond = (sma5 < sma10) & (sma5.shift(1) >= sma10.shift(1))
    x_5 = xcn(crossunder_cond, 5) >= 1

    stoch27 = (close_col - low_col.rolling(27).min()) / (high_col.rolling(27).max() - low_col.rolling(27).min()) * 100
    x_6_inner = xsa(stoch27, 5, 1)
    x_6 = 3 * xsa(x_6_inner, 3, 1) - 2 * xsa(xsa(stoch27, 5, 1), 3, 1)

    god_hunter = x_4 & x_5
    buy_trial = x_6 <= 13

    trend = godTrend
    xrf_x3_1 = pd.Series([xrf(x_3.iloc[max(0, i-9):i+1], 1) if i >= 1 else np.nan for i in range(len(x_3))], index=x_3.index)
    x_29 = x_3 > xrf_x3_1
    x_30 = x_3 <= xrf_x3_1
    ghLong = x_29 & (~x_29.shift(1).fillna(False))
    ghShort = x_30 & (~x_30.shift(1).fillna(False))

    buy_now = (x_6 > 5) & (x_6.shift(1) <= 5)
    buy_secretly = x_6 <= 5

    longCond = isUpTrend & ((ghLong) | ((trend < -120) & (buy_now | buy_secretly | buy_trial))) & filteredBuySignal
    shortCond = isDownTrend & ((ghShort) | (trend > 120)) & filteredSellSignal

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i < 5:
            continue
        cci_vals = [cciPeriod6.iloc[i], cciPeriod10.iloc[i], cciPeriod20.iloc[i], cciPeriod30.iloc[i], cciPeriod60.iloc[i]]
        if any(np.isnan(v) for v in cci_vals):
            continue
        if np.isnan(weightedMa.iloc[i]) or np.isnan(midTermMa.iloc[i]):
            continue
        if np.isnan(jLineMomentum.iloc[i]) or np.isnan(kLineFast.iloc[i]) or np.isnan(dLineSlow.iloc[i]):
            continue
        if np.isnan(x_1.iloc[i]) or np.isnan(x_2.iloc[i]) or np.isnan(x_3.iloc[i]):
            continue
        if np.isnan(x_6.iloc[i]):
            continue

        if longCond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
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

        if shortCond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
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