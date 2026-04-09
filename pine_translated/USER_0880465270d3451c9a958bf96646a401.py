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
    pi = 2 * np.arcsin(1)

    # E2PSS Parameters
    period_e2pss = 15
    a1 = np.exp(-1.414 * np.pi / period_e2pss)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / period_e2pss)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3

    n = len(df)
    filt2 = np.zeros(n)
    trigger_e2pss = np.zeros(n)

    # Calculate Filt2 and Trigger_E2PSS iteratively
    for i in range(n):
        price = df['close'].iloc[i]
        if i < 3:
            filt2[i] = price
        else:
            filt2[i] = coef1 * price + coef2 * filt2[i-1] + coef3 * filt2[i-2]
        trigger_e2pss[i] = filt2[i-1] if i > 0 else price

    filt2_series = pd.Series(filt2, index=df.index)
    trigger_e2pss_series = pd.Series(trigger_e2pss, index=df.index)

    # Signal conditions for E2PSS
    signal_long_e2pss = filt2_series > trigger_e2pss_series
    signal_short_e2pss = filt2_series < trigger_e2pss_series

    signal_long_e2pss_final = signal_long_e2pss
    signal_short_e2pss_final = signal_short_e2pss

    # Trendilo Parameters
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0

    # Trendilo Implementation
    close = df['close']
    pct_change = close.diff(trendilo_smooth) / close * 100

    def alma(arr, length, offset, sigma):
        w = np.exp(-np.square(np.arange(length) - offset * (length - 1)) / (2 * sigma * sigma))
        w = w / w.sum()
        return pd.Series(arr).rolling(length, min_periods=length).apply(lambda x: np.dot(x, w), raw=True)

    avg_pct_change = alma(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    rms = trendilo_bmult * np.sqrt((avg_pct_change ** 2).rolling(trendilo_length, min_periods=trendilo_length).mean())

    trendilo_dir = pd.Series(0, index=df.index)
    trendilo_dir[avg_pct_change > rms] = 1
    trendilo_dir[avg_pct_change < -rms] = -1

    # On-Chain Filter Parameters
    use_onchain = True
    onchain_volume_length = 20
    onchain_volume_threshold = 1.5
    onchain_delta_length = 14
    onchain_use_cvd = True
    onchain_use_vwap = True

    volume_ma = df['volume'].rolling(onchain_volume_length, min_periods=onchain_volume_length).mean()
    high_volume = df['volume'] > volume_ma * onchain_volume_threshold

    buy_volume = df['volume'] * ((df['close'] - df['low']) / (df['high'] - df['low']))
    sell_volume = df['volume'] * ((df['high'] - df['close']) / (df['high'] - df['low']))
    volume_delta = buy_volume - sell_volume

    cvd = volume_delta.cumsum()
    cvd_ma = cvd.rolling(onchain_delta_length, min_periods=onchain_delta_length).mean()
    cvd_trending_up = cvd > cvd_ma
    cvd_trending_down = cvd < cvd_ma

    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    price_above_vwap = df['close'] > vwap
    price_below_vwap = df['close'] < vwap

    mfi_length = 14
    mfi = pd.Series(0.0, index=df.index)
    for i in range(mfi_length, n):
        typical_prices = typical_price.iloc[i-mfi_length+1:i+1]
        raw_money_flow = typical_prices * df['volume'].iloc[i-mfi_length+1:i+1]
        positive_flow = raw_money_flow.where(typical_prices.diff() > 0, 0).sum()
        negative_flow = raw_money_flow.where(typical_prices.diff() < 0, 0).sum()
        if negative_flow != 0:
            money_flow_ratio = positive_flow / negative_flow
            mfi.iloc[i] = 100 - (100 / (1 + money_flow_ratio))
        else:
            mfi.iloc[i] = 100

    mfi_bullish = (mfi > 50) & (mfi < 80)
    mfi_bearish = (mfi < 50) & (mfi > 20)

    onchain_long_signal = pd.Series(True, index=df.index)
    onchain_short_signal = pd.Series(True, index=df.index)

    if use_onchain:
        onchain_long_signal = high_volume & cvd_trending_up & price_above_vwap & mfi_bullish
        onchain_short_signal = high_volume & cvd_trending_down & price_below_vwap & mfi_bearish

    # TTM Squeeze Parameters
    length_ttms = 20
    bb_mult_ttms = 2.0

    bb_basis = df['close'].rolling(length_ttms, min_periods=length_ttms).mean()
    bb_std = df['close'].rolling(length_ttms, min_periods=length_ttms).std()
    bb_upper = bb_basis + bb_mult_ttms * bb_std
    bb_lower = bb_basis - bb_mult_ttms * bb_std

    kc_basis = df['close'].rolling(length_ttms, min_periods=length_ttms).mean()
    tr = df['high'] - df['low']
    kc_dev = tr.rolling(length_ttms, min_periods=length_ttms).mean()

    kc_mult_high = 1.0
    kc_mult_mid = 1.5
    kc_mult_low = 2.0

    kc_upper_high = kc_basis + kc_dev * kc_mult_high
    kc_lower_high = kc_basis - kc_dev * kc_mult_high
    kc_upper_mid = kc_basis + kc_dev * kc_mult_mid
    kc_lower_mid = kc_basis - kc_dev * kc_mult_mid
    kc_upper_low = kc_basis + kc_dev * kc_mult_low
    kc_lower_low = kc_basis - kc_dev * kc_mult_low

    no_sqz_ttms = (bb_lower < kc_lower_low) | (bb_upper > kc_upper_low)
    low_sqz_ttms = (bb_lower >= kc_lower_low) | (bb_upper <= kc_upper_low)
    mid_sqz_ttms = (bb_lower >= kc_lower_mid) | (bb_upper <= kc_upper_mid)
    high_sqz_ttms = (bb_lower >= kc_lower_high) | (bb_upper <= kc_upper_high)

    highest_high = df['high'].rolling(length_ttms, min_periods=length_ttms).max()
    lowest_low = df['low'].rolling(length_ttms, min_periods=length_ttms).min()
    sma_close_ttms = df['close'].rolling(length_ttms, min_periods=length_ttms).mean()
    avg_hl = (highest_high + lowest_low) / 2.0
    linreg_input = df['close'] - (avg_hl + sma_close_ttms) / 2.0

    def linreg(series, length):
        result = pd.Series(np.nan, index=series.index)
        for i in range(length - 1, len(series)):
            y = series.iloc[i-length+1:i+1].values
            x = np.arange(length)
            x_mean = x.mean()
            y_mean = y.mean()
            slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
            intercept = y_mean - slope * x_mean
            result.iloc[i] = slope * (length - 1) + intercept
        return result

    mom_ttms = linreg(linreg_input, length_ttms)

    iff_1_ttms = pd.Series(1, index=df.index)
    iff_2_ttms = pd.Series(-1, index=df.index)

    iff_1_ttms[mom_ttms <= mom_ttms.shift(1)] = 2
    iff_2_ttms[mom_ttms >= mom_ttms.shift(1)] = -2

    ttms_signals_ttms = pd.Series(0, index=df.index)
    ttms_signals_ttms[mom_ttms > 0] = iff_1_ttms[mom_ttms > 0]
    ttms_signals_ttms[mom_ttms < 0] = iff_2_ttms[mom_ttms < 0]

    red_green_ttms = True
    basic_long_condition_ttms = (ttms_signals_ttms == 1) if red_green_ttms else (ttms_signals_ttms > 0)
    basic_short_condition_ttms = (ttms_signals_ttms == -1) if red_green_ttms else (ttms_signals_ttms < 0)

    highlight_movements_ttms = True
    ttms_signals_long_ttms = no_sqz_ttms & basic_long_condition_ttms if highlight_movements_ttms else basic_long_condition_ttms
    ttms_signals_short_ttms = no_sqz_ttms & basic_short_condition_ttms if highlight_movements_ttms else basic_short_condition_ttms

    cross_ttms = True
    ttms_signals_long_cross = (~ttms_signals_long_ttms.shift(1).fillna(False)) & ttms_signals_long_ttms if cross_ttms else ttms_signals_long_ttms
    ttms_signals_short_cross = (~ttms_signals_short_ttms.shift(1).fillna(False)) & ttms_signals_short_ttms if cross_ttms else ttms_signals_short_ttms

    use_ttms = True
    inverse_ttms = False
    ttms_signals_long_final = ttms_signals_short_cross if inverse_ttms else ttms_signals_long_cross if use_ttms else pd.Series(True, index=df.index)
    ttms_signals_short_final = ttms_signals_long_cross if inverse_ttms else ttms_signals_short_cross if use_ttms else pd.Series(True, index=df.index)

    # Final entry conditions
    long_condition = signal_long_e2pss_final & (trendilo_dir == 1) & basic_long_condition_ttms & onchain_long_signal
    short_condition = signal_short_e2pss_final & (trendilo_dir == -1) & basic_short_condition_ttms & onchain_short_signal

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(n):
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        entry_price = df['close'].iloc[i]

        if long_condition.iloc[i]:
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
        elif short_condition.iloc[i]:
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