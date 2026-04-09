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
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_vals = df['open'].values
    volume = df['volume'].values

    n = len(df)

    # ─── E2PSS Parameters ───
    useE2PSS = True
    inverseE2PSS = False
    PeriodE2PSS = 15

    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3

    PriceE2PSS = (high + low) / 2

    Filt2 = np.zeros(n)
    TriggerE2PSS = np.zeros(n)
    Filt2[0] = PriceE2PSS[0]
    Filt2[1] = PriceE2PSS[1]
    TriggerE2PSS[0] = PriceE2PSS[0]
    TriggerE2PSS[1] = PriceE2PSS[0]

    for i in range(2, n):
        Filt2[i] = coef1 * PriceE2PSS[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
        TriggerE2PSS[i] = Filt2[i-1]

    signalLongE2PSS = Filt2 > TriggerE2PSS
    signalShortE2PSS = Filt2 < TriggerE2PSS

    if inverseE2PSS:
        signalLongE2PSS_final = signalShortE2PSS
        signalShortE2PSS_final = signalLongE2PSS
    else:
        signalLongE2PSS_final = signalLongE2PSS
        signalShortE2PSS_final = signalShortE2PSS

    # ─── Trendilo Parameters ───
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0

    pct_change = np.zeros(n)
    for i in range(trendilo_smooth, n):
        pct_change[i] = (close[i] - close[i - trendilo_smooth]) / close[i] * 100

    def alma(arr, length, offset, sigma):
        m = np.arange(length)
        w = np.exp(-((m - offset * (length - 1)) ** 2) / (2 * (sigma ** 2)))
        w = w / w.sum()
        result = np.convolve(arr, w, mode='valid')
        pad = np.empty(length - 1)
        pad[:] = np.nan
        return np.concatenate([pad, result])

    avg_pct_change = alma(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)[:n]

    rms = np.zeros(n)
    for i in range(trendilo_length - 1, n):
        rms[i] = trendilo_bmult * np.sqrt(np.mean(avg_pct_change[i - trendilo_length + 1:i + 1] ** 2))

    trendilo_dir = np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0))

    # ─── TTMS Parameters ───
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    redGreen_TTMS = True
    cross_TTMS = True
    inverse_TTMS = False
    highlightMovements_TTMS = True

    KC_mult_high_TTMS = 1.0
    KC_mult_mid_TTMS = 1.5
    KC_mult_low_TTMS = 2.0

    BB_basis_TTMS = pd.Series(close).rolling(length_TTMS).mean().values
    close_std = pd.Series(close).rolling(length_TTMS).std().values
    dev_TTMS = BB_mult_TTMS * close_std
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS

    KC_basis_TTMS = BB_basis_TTMS.copy()
    tr1 = np.zeros(n)
    tr1[0] = high[0] - low[0]
    for i in range(1, n):
        tr1[i] = max(high[i] - low[i], max(abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
    devKC_TTMS = pd.Series(tr1).rolling(length_TTMS).mean().values

    KC_upper_high_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_high_TTMS
    KC_lower_high_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_high_TTMS
    KC_upper_mid_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_mid_TTMS
    KC_lower_mid_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_mid_TTMS
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS

    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    LowSqz_TTMS = (BB_lower_TTMS >= KC_lower_low_TTMS) | (BB_upper_TTMS <= KC_upper_low_TTMS)
    MidSqz_TTMS = (BB_lower_TTMS >= KC_lower_mid_TTMS) | (BB_upper_TTMS <= KC_upper_mid_TTMS)
    HighSqz_TTMS = (BB_lower_TTMS >= KC_lower_high_TTMS) | (BB_upper_TTMS <= KC_upper_high_TTMS)

    highest_high = pd.Series(high).rolling(length_TTMS).max().values
    lowest_low = pd.Series(low).rolling(length_TTMS).min().values
    sma_close = BB_basis_TTMS.copy()

    linreg_input = close - ((highest_high + lowest_low) / 2 + sma_close) / 2

    mom_TTMS = np.zeros(n)
    for i in range(length_TTMS - 1, n):
        x = np.arange(length_TTMS)
        y = linreg_input[i - length_TTMS + 1:i + 1]
        if len(y) == length_TTMS:
            y_mean = np.mean(y)
            slope = np.sum((x - np.mean(x)) * (y - y_mean)) / np.sum((x - np.mean(x)) ** 2)
            intercept = y_mean - slope * np.mean(x)
            mom_TTMS[i] = intercept

    mom_prev = np.roll(mom_TTMS, 1)
    mom_prev[0] = mom_TTMS[0]

    TTMS_Signals_TTMS = np.where(mom_TTMS > 0, 1, -1)

    basicLongCondition_TTMS = (TTMS_Signals_TTMS == 1) if redGreen_TTMS else (TTMS_Signals_TTMS > 0)
    basicShortCondition_TTMS = (TTMS_Signals_TTMS == -1) if redGreen_TTMS else (TTMS_Signals_TTMS < 0)

    TTMS_SignalsLong_TTMS = NoSqz_TTMS & basicLongCondition_TTMS if highlightMovements_TTMS else basicLongCondition_TTMS
    TTMS_SignalsShort_TTMS = NoSqz_TTMS & basicShortCondition_TTMS if highlightMovements_TTMS else basicShortCondition_TTMS

    TTMS_SignalsLong_TTMS_prev = np.roll(TTMS_SignalsLong_TTMS.astype(int), 1).astype(bool)
    TTMS_SignalsLong_TTMS_prev[0] = TTMS_SignalsLong_TTMS[0]
    TTMS_SignalsShort_TTMS_prev = np.roll(TTMS_SignalsShort_TTMS.astype(int), 1).astype(bool)
    TTMS_SignalsShort_TTMS_prev[0] = TTMS_SignalsShort_TTMS[0]

    TTMS_SignalsLongCross_TTMS = (~TTMS_SignalsLong_TTMS_prev) & TTMS_SignalsLong_TTMS if cross_TTMS else TTMS_SignalsLong_TTMS
    TTMS_SignalsShortCross_TTMS = (~TTMS_SignalsShort_TTMS_prev) & TTMS_SignalsShort_TTMS if cross_TTMS else TTMS_SignalsShort_TTMS

    if inverse_TTMS:
        TTMS_SignalsLongFinal_TTMS = TTMS_SignalsShortCross_TTMS
        TTMS_SignalsShortFinal_TTMS = TTMS_SignalsLongCross_TTMS
    else:
        TTMS_SignalsLongFinal_TTMS = TTMS_SignalsLongCross_TTMS
        TTMS_SignalsShortFinal_TTMS = TTMS_SignalsShortCross_TTMS

    # ─── Entry Conditions ───
    long_condition = signalLongE2PSS_final & (trendilo_dir == 1) & basicLongCondition_TTMS
    short_condition = signalShortE2PSS_final & (trendilo_dir == -1) & basicShortCondition_TTMS

    # ─── Generate Entries ───
    entries = []
    trade_num = 1
    in_position = False

    for i in range(1, n):
        if np.isnan(Filt2[i]) or np.isnan(avg_pct_change[i]) or np.isnan(mom_TTMS[i]):
            continue

        if not in_position:
            if long_condition.iloc[i] if hasattr(long_condition, 'iloc') else long_condition[i]:
                ts = int(df['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(close[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(close[i]),
                    'raw_price_b': float(close[i])
                })
                trade_num += 1
                in_position = True
            elif short_condition.iloc[i] if hasattr(short_condition, 'iloc') else short_condition[i]:
                ts = int(df['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(close[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(close[i]),
                    'raw_price_b': float(close[i])
                })
                trade_num += 1
                in_position = True

    return entries