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
    open_ohlc = df['open']

    # === E2PSS INDICATOR ===
    PeriodE2PSS = 15
    price_e2pss = (df['high'] + df['low']) / 2
    pi = 2 * np.arcsin(1)

    a1 = np.exp(-1.414 * np.pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3

    filt2 = np.zeros(len(df))
    trigger_e2pss = np.zeros(len(df))
    for i in range(len(df)):
        if i < 2:
            filt2[i] = price_e2pss.iloc[i]
        else:
            filt2[i] = coef1 * price_e2pss.iloc[i] + coef2 * filt2[i-1] + coef3 * filt2[i-2]
        if i > 0:
            trigger_e2pss[i] = filt2[i-1]
        else:
            trigger_e2pss[i] = price_e2pss.iloc[i]

    e2pss_long = filt2 > trigger_e2pss
    e2pss_short = filt2 < trigger_e2pss
    e2pss_long_final = e2pss_long
    e2pss_short_final = e2pss_short

    # === TRENDILO INDICATOR ===
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0

    pct_change = close.diff(trendilo_smooth) / close * 100
    avg_pct_change = pct_change.ewm(span=trendilo_length, adjust=False).mean()
    rms = trendilo_bmult * np.sqrt((avg_pct_change ** 2).rolling(trendilo_length).mean())

    trendilo_dir = np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0))
    trendilo_dir_series = pd.Series(trendilo_dir, index=df.index)
    trendilo_long = trendilo_dir_series == 1
    trendilo_short = trendilo_dir_series == -1

    # === EMA CLOUD INDICATOR ===
    ema_fast_len = 8
    ema_slow_len = 21
    ema_fast = close.ewm(span=ema_fast_len, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_len, adjust=False).mean()
    ema_cloud_long = ema_fast > ema_slow
    ema_cloud_short = ema_fast < ema_slow

    # === MAGIC TC PULLBACK INDICATOR ===
    mtcp_len = 20
    mtcp_atr_mult = 1.5

    def wilder_rsi(series, length):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def wilder_atr(length):
        high_low = high - low
        high_close = np.abs(high - close.shift(1))
        low_close = np.abs(low - close.shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr

    mtcp_basis = close.ewm(span=mtcp_len, adjust=False).mean()
    mtcp_atr = wilder_atr(mtcp_len)
    mtcp_upper = mtcp_basis + mtcp_atr * mtcp_atr_mult
    mtcp_lower = mtcp_basis - mtcp_atr * mtcp_atr_mult

    mtcp_basis_prev = mtcp_basis.shift(1)
    mtcp_trend_long = (close > mtcp_basis) & (mtcp_basis > mtcp_basis_prev)
    mtcp_trend_short = (close < mtcp_basis) & (mtcp_basis < mtcp_basis_prev)

    mtcp_pb_long = mtcp_trend_long & (low <= mtcp_basis) & (close > mtcp_basis)
    mtcp_pb_short = mtcp_trend_short & (high >= mtcp_basis) & (close < mtcp_basis)

    mtcp_long = mtcp_trend_long
    mtcp_short = mtcp_trend_short

    # === HACOLT INDICATOR ===
    hacolt_len = 10
    ha_close = (df['open'] + high + low + close) / 4
    ha_open = pd.Series(np.nan, index=df.index)
    ha_open.iloc[0] = (df['open'].iloc[0] + close.iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2

    ha_diff = ha_close - ha_open
    hacolt_smooth = ha_diff.ewm(span=hacolt_len, adjust=False).mean()
    hacolt_long = hacolt_smooth > 0
    hacolt_short = hacolt_smooth < 0

    # === TTM SQUEEZE INDICATOR ===
    length_ttms = 20
    bb_mult_ttms = 2.0
    bb_basis_ttms = close.rolling(length_ttms).mean()
    dev_ttms = bb_mult_ttms * close.rolling(length_ttms).std()
    bb_upper_ttms = bb_basis_ttms + dev_ttms
    bb_lower_ttms = bb_basis_ttms - dev_ttms

    kc_mult_high_ttms = 1.0
    kc_mult_mid_ttms = 1.5
    kc_mult_low_ttms = 2.0
    kc_basis_ttms = close.rolling(length_ttms).mean()
    dev_kc_ttms = df['tr'].ewm(span=length_ttms, adjust=False).mean()
    kc_upper_high_ttms = kc_basis_ttms + dev_kc_ttms * kc_mult_high_ttms
    kc_lower_high_ttms = kc_basis_ttms - dev_kc_ttms * kc_mult_high_ttms
    kc_upper_mid_ttms = kc_basis_ttms + dev_kc_ttms * kc_mult_mid_ttms
    kc_lower_mid_ttms = kc_basis_ttms - dev_kc_ttms * kc_mult_mid_ttms
    kc_upper_low_ttms = kc_basis_ttms + dev_kc_ttms * kc_mult_low_ttms
    kc_lower_low_ttms = kc_basis_ttms - dev_kc_ttms * kc_mult_low_ttms

    nosqz_ttms = (bb_lower_ttms < kc_lower_low_ttms) | (bb_upper_ttms > kc_upper_low_ttms)

    mom_ttms = pd.Series(index=df.index, dtype=float)
    for i in range(length_ttms - 1, len(df)):
        highest_high = high.iloc[i-length_ttms+1:i+1].max()
        lowest_low = low.iloc[i-length_ttms+1:i+1].min()
        sma_close = close.iloc[i-length_ttms+1:i+1].mean()
        linreg_input = (close.iloc[i] - (highest_high + lowest_low) / 2) - sma_close
        x = np.arange(length_ttms)
        x_mean = x.mean()
        y_vals = close.iloc[i-length_ttms+1:i+1].values - ((highest_high + lowest_low) / 2 + sma_close)
        slope = np.sum((x - x_mean) * y_vals) / np.sum((x - x_mean) ** 2)
        intercept = y_vals.mean() - slope * x_mean
        mom_ttms.iloc[i] = slope * (length_ttms - 1) + intercept

    mom_ttms_prev = mom_ttms.shift(1)
    ttms_signals = pd.Series(np.where(mom_ttms > 0, np.where(mom_ttms > mom_ttms_prev, 1, 2), np.where(mom_ttms < mom_ttms_prev, -1, -2)), index=df.index)

    basic_long_cond_ttms = ttms_signals == 1
    basic_short_cond_ttms = ttms_signals == -1

    ttms_signals_long = nosqz_ttms & basic_long_cond_ttms
    ttms_signals_short = nosqz_ttms & basic_short_cond_ttms

    ttms_signals_long_cross = ttms_signals_long & ~(ttms_signals_long.shift(1).fillna(False))
    ttms_signals_short_cross = ttms_signals_short & ~(ttms_signals_short.shift(1).fillna(False))

    ttms_signals_long_final = ttms_signals_long_cross
    ttms_signals_short_final = ttms_signals_short_cross

    # === SIGNAL ROUTING (defaults per comments) ===
    baseline_source = "E2PSS"
    confirmation_source = "Trendilo"
    use_ttm_filter = True

    baseline_long = e2pss_long_final
    baseline_short = e2pss_short_final

    confirmation_long = trendilo_long
    confirmation_short = trendilo_short

    # Combined long signal
    long_signal = baseline_long & confirmation_long
    if use_ttm_filter:
        long_signal = long_signal & ttms_signals_long_final

    # Combined short signal
    short_signal = baseline_short & confirmation_short
    if use_ttm_filter:
        short_signal = short_signal & ttms_signals_short_final

    # === GENERATE ENTRIES ===
    entries = []
    trade_num = 1

    for i in range(1, len(df)):
        if pd.isna(filt2[i]) or pd.isna(trigger_e2pss[i]):
            continue
        if pd.isna(avg_pct_change.iloc[i]) or pd.isna(rms.iloc[i]):
            continue
        if pd.isna(mtcp_basis.iloc[i]) or pd.isna(mtcp_atr.iloc[i]):
            continue

        entry_price = close.iloc[i]
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        if long_signal.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_signal.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
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