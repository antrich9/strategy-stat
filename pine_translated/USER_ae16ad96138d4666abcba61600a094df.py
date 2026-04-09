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

    # ========== QQE MOD (First) ==========
    RSI_Period = 6
    SF = 6
    QQE = 3
    ThreshHold = 3
    Wilders_Period = RSI_Period * 2 - 1

    def wilder_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    Rsi = wilder_rsi(close, RSI_Period)
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    AtrRsi = (RsiMa.shift(1) - RsiMa).abs()
    MaAtrRsi = AtrRsi.ewm(alpha=1/Wilders_Period, min_periods=Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi.ewm(alpha=1/Wilders_Period, min_periods=Wilders_Period, adjust=False).mean() * QQE

    RSIndex = RsiMa
    longband = pd.Series(0.0, index=df.index)
    shortband = pd.Series(0.0, index=df.index)
    trend = pd.Series(0, index=df.index)
    FastAtrRsiTL = pd.Series(0.0, index=df.index)

    for i in range(1, len(df)):
        newshortband_val = RSIndex.iloc[i] + dar.iloc[i]
        newlongband_val = RSIndex.iloc[i] - dar.iloc[i]
        if RSIndex.iloc[i-1] > longband.iloc[i-1] and RSIndex.iloc[i] > longband.iloc[i-1]:
            longband.iloc[i] = max(longband.iloc[i-1], newlongband_val)
        else:
            longband.iloc[i] = newlongband_val
        if RSIndex.iloc[i-1] < shortband.iloc[i-1] and RSIndex.iloc[i] < shortband.iloc[i-1]:
            shortband.iloc[i] = min(shortband.iloc[i-1], newshortband_val)
        else:
            shortband.iloc[i] = newshortband_val
        cross_1 = longband.iloc[i-1] < RSIndex.iloc[i] and longband.iloc[i-1] >= RSIndex.iloc[i-1]
        if RSIndex.iloc[i] < shortband.iloc[i-1]:
            trend.iloc[i] = 1
        elif cross_1:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
        FastAtrRsiTL.iloc[i] = longband.iloc[i] if trend.iloc[i] == 1 else shortband.iloc[i]

    # Bollinger on QQE
    QQE_len = 50
    qqeMult = 0.35
    basis = (FastAtrRsiTL - 50).rolling(QQE_len).mean()
    dev = qqeMult * (FastAtrRsiTL - 50).rolling(QQE_len).std()
    upper_bb = basis + dev
    lower_bb = basis - dev

    # ========== QQE MOD (Second) ==========
    RSI_Period2 = 6
    SF2 = 5
    QQE2 = 1.61
    ThreshHold2 = 3
    Wilders_Period2 = RSI_Period2 * 2 - 1

    Rsi2 = wilder_rsi(close, RSI_Period2)
    RsiMa2 = Rsi2.ewm(span=SF2, adjust=False).mean()
    AtrRsi2 = (RsiMa2.shift(1) - RsiMa2).abs()
    MaAtrRsi2 = AtrRsi2.ewm(alpha=1/Wilders_Period2, min_periods=Wilders_Period2, adjust=False).mean()
    dar2 = MaAtrRsi2.ewm(alpha=1/Wilders_Period2, min_periods=Wilders_Period2, adjust=False).mean() * QQE2

    RSIndex2 = RsiMa2
    longband2 = pd.Series(0.0, index=df.index)
    shortband2 = pd.Series(0.0, index=df.index)
    trend2 = pd.Series(0, index=df.index)
    FastAtrRsi2TL = pd.Series(0.0, index=df.index)

    for i in range(1, len(df)):
        newshortband2_val = RSIndex2.iloc[i] + dar2.iloc[i]
        newlongband2_val = RSIndex2.iloc[i] - dar2.iloc[i]
        if RSIndex2.iloc[i-1] > longband2.iloc[i-1] and RSIndex2.iloc[i] > longband2.iloc[i-1]:
            longband2.iloc[i] = max(longband2.iloc[i-1], newlongband2_val)
        else:
            longband2.iloc[i] = newlongband2_val
        if RSIndex2.iloc[i-1] < shortband2.iloc[i-1] and RSIndex2.iloc[i] < shortband2.iloc[i-1]:
            shortband2.iloc[i] = min(shortband2.iloc[i-1], newshortband2_val)
        else:
            shortband2.iloc[i] = newshortband2_val
        cross_2 = longband2.iloc[i-1] < RSIndex2.iloc[i] and longband2.iloc[i-1] >= RSIndex2.iloc[i-1]
        if RSIndex2.iloc[i] < shortband2.iloc[i-1]:
            trend2.iloc[i] = 1
        elif cross_2:
            trend2.iloc[i] = -1
        else:
            trend2.iloc[i] = trend2.iloc[i-1]
        FastAtrRsi2TL.iloc[i] = longband2.iloc[i] if trend2.iloc[i] == 1 else shortband2.iloc[i]

    # Greenbar/Redbar conditions
    Greenbar1 = RsiMa2 - 50 > ThreshHold2
    Greenbar2 = RsiMa - 50 > upper_bb
    Redbar1 = RsiMa2 - 50 < -ThreshHold2
    Redbar2 = RsiMa - 50 < lower_bb

    # ========== SSL Hybrid ==========
    atrlen = 14
    mult = 1.0

    def wma(s, length):
        weights = np.arange(1, length + 1)
        return s.rolling(length).apply(lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum() if len(x) == length else np.nan, raw=True)

    atr_slen = wma((high - low).clip(lower=0), atrlen)

    # HMA function
    def hma(src, length):
        return wma(2 * wma(src, length // 2) - wma(src, length), int(np.sqrt(length)))

    # JMA approximation (simplified)
    def jma(src, length, phase=3, power=1):
        alpha = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
        jma_val = pd.Series(0.0, index=src.index)
        for i in range(1, len(src)):
            jma_val.iloc[i] = alpha * src.iloc[i] + (1 - alpha) * jma_val.iloc[i-1]
        return jma_val

    ssl1_len = 60
    ssl2_len = 5
    ssl3_len = 15

    SSL1 = hma(close, ssl1_len)
    SSL2 = jma(close, ssl2_len)
    SSL3 = hma(close, ssl3_len)

    # Baseline and SSL lines
    Baseline = SSL1
    SSL1_line = SSL2
    SSL2_line = SSL3

    # ========== Entry Conditions ==========
    # QQE Long: Greenbar1 and Greenbar2
    qqe_long_cond = Greenbar1 & Greenbar2

    # QQE Short: Redbar1 and Redbar2
    qqe_short_cond = Redbar1 & Redbar2

    # SSL Hybrid long: close above Baseline and close above SSL1_line
    ssl_long_cond = (close > Baseline) & (close > SSL1_line)

    # SSL Hybrid short: close below Baseline and close below SSL1_line
    ssl_short_cond = (close < Baseline) & (close < SSL1_line)

    # Combined: QQE confirms AND SSL confirms
    long_entry = qqe_long_cond & ssl_long_cond
    short_entry = qqe_short_cond & ssl_short_cond

    # ========== Generate Entries ==========
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if long_entry.iloc[i] and not long_entry.iloc[i-1] if i > 0 else False:
            entry_price = close.iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_entry.iloc[i] and not short_entry.iloc[i-1] if i > 0 else False:
            entry_price = close.iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries