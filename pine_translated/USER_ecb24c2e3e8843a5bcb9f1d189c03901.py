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
    results = []
    trade_num = 1

    close = df['close']
    high = df['high']
    low = df['low']

    # E2PSS Parameters (useE2PSS=true, inverseE2PSS=false)
    PeriodE2PSS = 15
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * np.pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3

    PriceE2PSS = (high + low) / 2

    Filt2 = np.zeros(len(df))
    TriggerE2PSS = np.zeros(len(df))
    Filt2[0] = PriceE2PSS.iloc[0]
    Filt2[1] = PriceE2PSS.iloc[1]
    TriggerE2PSS[0] = PriceE2PSS.iloc[0]

    for i in range(2, len(df)):
        Filt2[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
        TriggerE2PSS[i] = Filt2[i-1]

    signalLongE2PSS = Filt2 > TriggerE2PSS
    signalShortE2PSS = Filt2 < TriggerE2PSS

    signalLongE2PSSFinal = signalLongE2PSS
    signalShortE2PSSFinal = signalShortE2PSS

    # Trendilo
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0

    pct_change = (close.shift(-trendilo_smooth) - close) / close * 100
    avg_pct_change = pct_change.ewm(span=trendilo_length, adjust=False).mean()

    window_vals = avg_pct_change ** 2
    rms = trendilo_bmult * np.sqrt(window_vals.rolling(trendilo_length).sum() / trendilo_length)

    trendilo_dir = np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0))

    # TTM Squeeze
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = BB_mult_TTMS * close.rolling(length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS

    KC_mult_low_TTMS = 2.0
    KC_mult_mid_TTMS = 1.5
    KC_mult_high_TTMS = 1.0
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
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
    avg_of_hilo = (highest_high + lowest_low) / 2
    mom_TTMS_raw = close - avg_of_hilo - close.rolling(length_TTMS).mean()
    mom_TTMS = mom_TTMS_raw.rolling(length_TTMS).mean()

    mom_prev = mom_TTMS.shift(1)
    TTMS_Signals_TTMS = np.where(mom_TTMS > 0, np.where(mom_TTMS > mom_prev, 1, 2), np.where(mom_TTMS < mom_prev, -1, -2))

    redGreen_TTMS = True
    basicLongCondition_TTMS = np.where(redGreen_TTMS, TTMS_Signals_TTMS == 1, TTMS_Signals_TTMS > 0)
    basicShortCondition_TTMS = np.where(redGreen_TTMS, TTMS_Signals_TTMS == -1, TTMS_Signals_TTMS < 0)

    highlightMovements_TTMS = True
    TTMS_SignalsLong_TTMS = np.where(highlightMovements_TTMS, NoSqz_TTMS & (basicLongCondition_TTMS == True), basicLongCondition_TTMS == True)
    TTMS_SignalsShort_TTMS = np.where(highlightMovements_TTMS, NoSqz_TTMS & (basicShortCondition_TTMS == True), basicShortCondition_TTMS == True)

    cross_TTMS = True
    use_TTMS = True
    inverse_TTMS = False

    TTMS_SignalsLongCross_TTMS = np.where(cross_TTMS, (~TTMS_SignalsLong_TTMS.shift(1).fillna(False)) & TTMS_SignalsLong_TTMS, TTMS_SignalsLong_TTMS)
    TTMS_SignalsShortCross_TTMS = np.where(cross_TTMS, (~TTMS_SignalsShort_TTMS.shift(1).fillna(False)) & TTMS_SignalsShort_TTMS, TTMS_SignalsShort_TTMS)

    TTMS_SignalsLongFinal_TTMS = np.where(use_TTMS, np.where(inverse_TTMS, TTMS_SignalsShortCross_TTMS, TTMS_SignalsLongCross_TTMS), True)
    TTMS_SignalsShortFinal_TTMS = np.where(use_TTMS, np.where(inverse_TTMS, TTMS_SignalsLongCross_TTMS, TTMS_SignalsShortCross_TTMS), True)

    # emaBullish/emaBearish - using local EMA approximation since we don't have 240min TF data
    emaFast = close.ewm(span=10, adjust=False).mean()
    emaSlow = close.ewm(span=50, adjust=False).mean()
    emaBullish = emaFast > emaSlow
    emaBearish = emaFast < emaSlow

    long_condition = signalLongE2PSSFinal & (trendilo_dir == 1) & (basicLongCondition_TTMS == True) & emaBullish
    short_condition = signalShortE2PSSFinal & (trendilo_dir == -1) & (basicShortCondition_TTMS == True) & emaBearish

    pos_size = 0
    for i in range(1, len(df)):
        if pos_size == 0:
            if long_condition.iloc[i]:
                entry_price = close.iloc[i]
                results.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
                pos_size = 1
            elif short_condition.iloc[i]:
                entry_price = close.iloc[i]
                results.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': int(df['time'].iloc[i]),
                    'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
                pos_size = -1

    return results