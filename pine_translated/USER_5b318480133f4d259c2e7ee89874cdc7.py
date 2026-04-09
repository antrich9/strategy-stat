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
    open_price = df['open']
    volume = df['volume']

    useE2PSS = True
    inverseE2PSS = False
    PeriodE2PSS = 15

    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0

    use_TTMS = True
    redGreen_TTMS = True
    cross_TTMS = True
    inverse_TTMS = False
    highlightMovements_TTMS = True
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    KC_mult_high_TTMS = 1.0
    KC_mult_mid_TTMS = 1.5
    KC_mult_low_TTMS = 2.0

    pi = 2 * np.arcsin(1)
    Period = PeriodE2PSS
    a1 = np.exp(-1.414 * np.pi / Period)
    b1 = 2 * a1 * np.cos(1.414 * pi / Period)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3

    Filt2 = np.zeros(len(df))
    Filt2[0] = close.iloc[0]
    Filt2[1] = close.iloc[1]
    for i in range(2, len(df)):
        Filt2[i] = coef1 * close.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]

    TriggerE2PSS = np.roll(Filt2, 1)
    TriggerE2PSS[0] = Filt2[0]

    signalLongE2PSS_raw = Filt2 > TriggerE2PSS
    signalShortE2PSS_raw = Filt2 < TriggerE2PSS
    signalLongE2PSS = signalLongE2PSS_raw if useE2PSS else np.ones(len(df), dtype=bool)
    signalShortE2PSS = signalShortE2PSS_raw if useE2PSS else np.ones(len(df), dtype=bool)
    signalLongE2PSSFinal = signalShortE2PSS if inverseE2PSS else signalLongE2PSS
    signalShortE2PSSFinal = signalLongE2PSS if inverseE2PSS else signalShortE2PSS

    pct_change = close.diff(trendilo_smooth) / close * 100
    n = len(pct_change)
    offset_n = int(trendilo_offset * (trendilo_length - 1))
    sigma_inv = 2 * trendilo_sigma * trendilo_sigma
    denom_sum = 0
    for j in range(trendilo_length):
        w = np.exp(-((j - offset_n) ** 2) / sigma_inv)
        denom_sum += w
    alma_vals = np.zeros(n)
    for i in range(trendilo_length - 1, n):
        num_sum = 0.0
        for j in range(trendilo_length):
            w = np.exp(-((j - offset_n) ** 2) / sigma_inv)
            idx = i - trendilo_length + 1 + j
            if idx >= 0:
                num_sum += w * pct_change.iloc[idx]
        alma_vals[i] = num_sum / denom_sum
    avg_pct_change = pd.Series(alma_vals, index=pct_change.index)
    rms_vals = np.zeros(n)
    for i in range(trendilo_length - 1, n):
        window = avg_pct_change.iloc[i - trendilo_length + 1:i + 1].values
        rms_vals[i] = trendilo_bmult * np.sqrt(np.sum(window * window) / trendilo_length)
    rms = pd.Series(rms_vals, index=pct_change.index)
    trendilo_dir = np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0))
    trendilo_dir = pd.Series(trendilo_dir, index=close.index)

    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = BB_mult_TTMS * close.rolling(length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    devKC_TTMS = tr.rolling(length_TTMS).mean()
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

    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    median_avg = (highest_high + lowest_low + close.rolling(length_TTMS).mean()) / 3
    mom_TTMS = (close - median_avg).rolling(length_TTMS).apply(
        lambda x: np.polyfit(np.arange(length_TTMS), x, 1)[0] * (length_TTMS * (length_TTMS - 1) / 2), raw=True
    )

    mom_prev = mom_TTMS.shift(1)
    TTMS_Signals_TTMS = np.where(
        mom_TTMS > 0,
        np.where(mom_TTMS > mom_prev, 1, 2),
        np.where(mom_TTMS < mom_prev, -1, -2)
    )
    TTMS_Signals_TTMS = pd.Series(TTMS_Signals_TTMS, index=close.index)

    basicLongCondition_TTMS = TTMS_Signals_TTMS == 1 if redGreen_TTMS else TTMS_Signals_TTMS > 0
    basicShortCondition_TTMS = TTMS_Signals_TTMS == -1 if redGreen_TTMS else TTMS_Signals_TTMS < 0

    TTMS_SignalsLong_TTMS = np.where(
        highlightMovements_TTMS,
        NoSqz_TTMS & basicLongCondition_TTMS,
        basicLongCondition_TTMS
    )
    TTMS_SignalsShort_TTMS = np.where(
        highlightMovements_TTMS,
        NoSqz_TTMS & basicShortCondition_TTMS,
        basicShortCondition_TTMS
    )

    TTMS_SignalsLongCross_TTMS = pd.Series(False, index=close.index)
    TTMS_SignalsShortCross_TTMS = pd.Series(False, index=close.index)
    for i in range(1, len(close)):
        if TTMS_SignalsLong_TTMS.iloc[i] and not TTMS_SignalsLong_TTMS.iloc[i-1]:
            TTMS_SignalsLongCross_TTMS.iloc[i] = True
        if TTMS_SignalsShort_TTMS.iloc[i] and not TTMS_SignalsShort_TTMS.iloc[i-1]:
            TTMS_SignalsShortCross_TTMS.iloc[i] = True

    TTMS_SignalsLongFinal_TTMS = pd.Series(True, index=close.index)
    TTMS_SignalsShortFinal_TTMS = pd.Series(True, index=close.index)
    if use_TTMS:
        TTMS_SignalsLongFinal_TTMS = TTMS_SignalsShortCross_TTMS if inverse_TTMS else TTMS_SignalsLongCross_TTMS
        TTMS_SignalsShortFinal_TTMS = TTMS_SignalsLongCross_TTMS if inverse_TTMS else TTMS_SignalsShortCross_TTMS

    long_condition = signalLongE2PSSFinal & (trendilo_dir == 1) & (TTMS_SignalsLongFinal_TTMS != 0)
    short_condition = signalShortE2PSSFinal & (trendilo_dir == -1) & (TTMS_SignalsShortFinal_TTMS != 0)

    trade_num = 0
    entries = []
    in_position = False

    for i in range(1, len(close)):
        if np.isnan(TTMS_SignalsLongFinal_TTMS.iloc[i]):
            continue
        if np.isnan(trendilo_dir.iloc[i]):
            continue

        if long_condition.iloc[i] and not in_position:
            trade_num += 1
            in_position = True
            entry_price = close.iloc[i]
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
        elif short_condition.iloc[i] and not in_position:
            trade_num += 1
            in_position = True
            entry_price = close.iloc[i]
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })

    return entries