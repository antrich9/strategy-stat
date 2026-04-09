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
    # Helper function: Wilder RSI
    def wilders_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Helper function: Wilder ATR
    def wilders_atr(high, low, close, period):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/period, adjust=False).mean()
        return atr

    # Helper: Hull Moving Average
    def hma(src, length):
        half_len = int(length / 2)
        sqrt_len = int(np.sqrt(length))
        wma1 = src.rolling(half_len).apply(lambda x: np.dot(x, np.arange(1, half_len+1)) / np.arange(1, half_len+1).sum(), raw=True)
        wma2 = src.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1)) / np.arange(1, length+1).sum(), raw=True)
        diff = 2 * wma1 - wma2
        hma_result = diff.rolling(sqrt_len).apply(lambda x: np.dot(x, np.arange(1, sqrt_len+1)) / np.arange(1, sqrt_len+1).sum(), raw=True)
        return hma_result

    # Helper: EMA
    def ema_func(src, length):
        return src.ewm(span=length, adjust=False).mean()

    # Helper: SMA
    def sma_func(src, length):
        return src.rolling(length).mean()

    # Helper: DEMA
    def dema(src, length):
        e = ema_func(src, length)
        return 2 * e - ema_func(e, length)

    # Helper: TEMA
    def tema(src, length):
        e1 = ema_func(src, length)
        e2 = ema_func(e1, length)
        e3 = ema_func(e2, length)
        return 3 * e1 - 3 * e2 + e3

    # Helper: WMA
    def wma(src, length):
        weights = np.arange(1, length + 1)
        return src.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    # QQE MOD calculations (first)
    RSI_Period = 6
    SF = 6
    QQE_factor = 3
    Wilders_Period = RSI_Period * 2 - 1

    rsi_1 = wilders_rsi(df['close'], RSI_Period)
    rsi_ma_1 = ema_func(rsi_1, SF)
    atr_rsi_1 = (rsi_ma_1.shift(1) - rsi_ma_1).abs()
    ma_atr_rsi_1 = ema_func(atr_rsi_1, Wilders_Period)
    dar_1 = ema_func(ma_atr_rsi_1, Wilders_Period) * QQE_factor

    longband_1 = pd.Series(0.0, index=df.index)
    shortband_1 = pd.Series(0.0, index=df.index)
    trend_1 = pd.Series(0, index=df.index)

    for i in range(2, len(df)):
        rs_idx = rsi_ma_1.iloc[i]
        rs_idx_prev = rsi_ma_1.iloc[i-1]
        dar_val = dar_1.iloc[i]
        lb_prev = longband_1.iloc[i-1]
        sb_prev = shortband_1.iloc[i-1]

        new_short = rs_idx + dar_val
        new_long = rs_idx - dar_val

        if rs_idx_prev > lb_prev and rs_idx > lb_prev:
            new_longband = max(lb_prev, new_long)
        else:
            new_longband = new_long

        if rs_idx_prev < sb_prev and rs_idx < sb_prev:
            new_shortband = min(sb_prev, new_short)
        else:
            new_shortband = new_short

        longband_1.iloc[i] = new_longband
        shortband_1.iloc[i] = new_shortband

        cross_up = longband_1.iloc[i-1] < rs_idx_prev and rs_idx > longband_1.iloc[i-1]
        cross_down = rs_idx < shortband_1.iloc[i-1] and rs_idx_prev > shortband_1.iloc[i-1]

        if cross_down:
            trend_1.iloc[i] = 1
        elif cross_up:
            trend_1.iloc[i] = -1
        else:
            trend_1.iloc[i] = trend_1.iloc[i-1] if not pd.isna(trend_1.iloc[i-1]) else 0

    fast_atr_rsi_1 = pd.Series(0.0, index=df.index)
    for i in range(len(df)):
        if trend_1.iloc[i] == 1:
            fast_atr_rsi_1.iloc[i] = longband_1.iloc[i]
        else:
            fast_atr_rsi_1.iloc[i] = shortband_1.iloc[i]

    # Bollinger Bands on QQE1
    qqe1_centered = fast_atr_rsi_1 - 50
    bb_basis_1 = sma_func(qqe1_centered, 50)
    bb_std_1 = qqe1_centered.rolling(50).std()
    qqe_upper_1 = bb_basis_1 + 0.35 * bb_std_1
    qqe_lower_1 = bb_basis_1 - 0.35 * bb_std_1

    # QQE MOD calculations (second)
    RSI_Period2 = 6
    SF2 = 5
    QQE2_factor = 1.61
    Wilders_Period2 = RSI_Period2 * 2 - 1

    rsi_2 = wilders_rsi(df['close'], RSI_Period2)
    rsi_ma_2 = ema_func(rsi_2, SF2)
    atr_rsi_2 = (rsi_ma_2.shift(1) - rsi_ma_2).abs()
    ma_atr_rsi_2 = ema_func(atr_rsi_2, Wilders_Period2)
    dar_2 = ema_func(ma_atr_rsi_2, Wilders_Period2) * QQE2_factor

    longband_2 = pd.Series(0.0, index=df.index)
    shortband_2 = pd.Series(0.0, index=df.index)
    trend_2 = pd.Series(0, index=df.index)

    for i in range(2, len(df)):
        rs_idx = rsi_ma_2.iloc[i]
        rs_idx_prev = rsi_ma_2.iloc[i-1]
        dar_val = dar_2.iloc[i]
        lb_prev = longband_2.iloc[i-1]
        sb_prev = shortband_2.iloc[i-1]

        new_short = rs_idx + dar_val
        new_long = rs_idx - dar_val

        if rs_idx_prev > lb_prev and rs_idx > lb_prev:
            new_longband = max(lb_prev, new_long)
        else:
            new_longband = new_long

        if rs_idx_prev < sb_prev and rs_idx < sb_prev:
            new_shortband = min(sb_prev, new_short)
        else:
            new_shortband = new_short

        longband_2.iloc[i] = new_longband
        shortband_2.iloc[i] = new_shortband

        cross_up = longband_2.iloc[i-1] < rs_idx_prev and rs_idx > longband_2.iloc[i-1]
        cross_down = rs_idx < shortband_2.iloc[i-1] and rs_idx_prev > shortband_2.iloc[i-1]

        if cross_down:
            trend_2.iloc[i] = 1
        elif cross_up:
            trend_2.iloc[i] = -1
        else:
            trend_2.iloc[i] = trend_2.iloc[i-1] if not pd.isna(trend_2.iloc[i-1]) else 0

    fast_atr_rsi_2 = pd.Series(0.0, index=df.index)
    for i in range(len(df)):
        if trend_2.iloc[i] == 1:
            fast_atr_rsi_2.iloc[i] = longband_2.iloc[i]
        else:
            fast_atr_rsi_2.iloc[i] = shortband_2.iloc[i]

    # QQE bar conditions
    threshold1 = 3
    threshold2 = 3

    greenbar1 = rsi_ma_2 - 50 > threshold2
    greenbar2 = rsi_ma_1 - 50 > qqe_upper_1
    redbar1 = rsi_ma_2 - 50 < -threshold2
    redbar2 = rsi_ma_1 - 50 < qqe_lower_1

    # SSL Hybrid
    ssl_baseline = hma(df['close'], 60)

    # ATR for SSL
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_ssl = ema_func(tr, 14)
    ssl_upper = df['close'] + atr_ssl
    ssl_lower = df['close'] - atr_ssl

    # SSL2 (DEMA)
    ssl2_val = dema(df['close'], 5)

    # Waddah Attar Explosion
    fast_ema = ema_func(df['close'], 9)
    slow_ema = ema_func(df['close'], 21)
    wae_diff = (fast_ema - slow_ema).abs()
    wae_explosion = wae_diff * ema_func(tr, 150) * 150

    # Entry conditions
    long_cond = greenbar1 & greenbar2 & (wae_explosion > 0) & (df['close'] > ssl_baseline)
    short_cond = redbar1 & redbar2 & (wae_explosion > 0) & (df['close'] < ssl_baseline)

    entries = []
    trade_num = 1

    for i in range(1, len(df)):
        if long_cond.iloc[i]:
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
        elif short_cond.iloc[i]:
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