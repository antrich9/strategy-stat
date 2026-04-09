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
    # Parameters matching Pine Script inputs
    useE2PSS = True
    inverseE2PSS = False
    PriceE2PSS = (df['high'] + df['low']) / 2
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

    # ─────────────────────────────────────────────────────────────────────────────
    # E2PSS (Two Pole Super Smooth Filter)
    # ─────────────────────────────────────────────────────────────────────────────
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * np.pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3

    Filt2 = np.zeros(len(df))
    TriggerE2PSS = np.zeros(len(df))
    Filt2[0] = PriceE2PSS.iloc[0]
    TriggerE2PSS[0] = PriceE2PSS.iloc[0]

    for i in range(1, len(df)):
        if i < 3:
            Filt2[i] = PriceE2PSS.iloc[i]
        else:
            Filt2[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
        TriggerE2PSS[i] = Filt2[i-1] if i > 0 else Filt2[0]

    Filt2 = pd.Series(Filt2, index=df.index)
    TriggerE2PSS = pd.Series(TriggerE2PSS, index=df.index)

    signalLongE2PSS = Filt2 > TriggerE2PSS
    signalShortE2PSS = Filt2 < TriggerE2PSS

    signalLongE2PSSFinal = ~inverseE2PSS & signalLongE2PSS if isinstance(inverseE2PSS, bool) else signalShortE2PSS
    signalShortE2PSSFinal = ~inverseE2PSS & signalShortE2PSS if isinstance(inverseE2PSS, bool) else signalLongE2PSS
    if inverseE2PSS:
        signalLongE2PSSFinal = signalShortE2PSS
        signalShortE2PSSFinal = signalLongE2PSS
    else:
        signalLongE2PSSFinal = signalLongE2PSS
        signalShortE2PSSFinal = signalShortE2PSS

    # ─────────────────────────────────────────────────────────────────────────────
    # Trendilo
    # ─────────────────────────────────────────────────────────────────────────────
    close = df['close']
    pct_change = close.pct_change(trendilo_smooth) * 100

    def alma_rolling(series, length, offset, sigma):
        m = int(np.floor(offset * (length - 1)))
        s = sigma * (length - 1) / 6
        weights = np.exp(-((np.arange(length) - m) ** 2) / (2 * s * s))
        weights = weights / weights.sum()

        result = pd.Series(index=series.index, dtype=float)
        for i in range(length - 1, len(series)):
            values = series.iloc[i - length + 1:i + 1].values
            result.iloc[i] = np.sum(values * weights)
        return result

    avg_pct_change = alma_rolling(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)

    rms_values = pd.Series(index=df.index, dtype=float)
    for i in range(trendilo_length - 1, len(df)):
        window = avg_pct_change.iloc[i - trendilo_length + 1:i + 1]
        rms_values.iloc[i] = trendilo_bmult * np.sqrt((window ** 2).mean())

    trendilo_dir = pd.Series(0, index=df.index)
    trendilo_dir = np.where(avg_pct_change > rms_values, 1, trendilo_dir)
    trendilo_dir = np.where(avg_pct_change < -rms_values, -1, trendilo_dir)
    trendilo_dir = pd.Series(trendilo_dir, index=df.index)

    # ─────────────────────────────────────────────────────────────────────────────
    # TTMS (TTM Squeeze)
    # ─────────────────────────────────────────────────────────────────────────────
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = BB_mult_TTMS * close.rolling(length_TTMS).std()

    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS

    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    tr = df['high'].combine_subtract(df['low'], fill_value=0)
    tr = tr.combine_max(df['high'].diff().abs().fillna(0), fill_value=0)
    tr = tr.combine_max((df['low'].diff() * -1).abs().fillna(0), fill_value=0)
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

    highest_high = df['high'].rolling(length_TTMS).max()
    lowest_low = df['low'].rolling(length_TTMS).min()
    avg_high_low_sma = ((highest_high + lowest_low) / 2 + close.rolling(length_TTMS).mean()) / 2
    mom_TTMS_raw = close - avg_high_low_sma

    mom_TTMS = mom_TTMS_raw.rolling(length_TTMS).mean()

    mom_prev = mom_TTMS.shift(1)
    iff_1_TTMS_no = (mom_TTMS > mom_prev).astype(int).replace(0, 2)
    iff_2_TTMS_no = (mom_TTMS < mom_prev).astype(int).replace(0, -2)

    TTMS_Signals_TTMS = pd.Series(0, index=df.index)
    TTMS_Signals_TTMS = np.where(mom_TTMS > 0, iff_1_TTMS_no, iff_2_TTMS_no)
    TTMS_Signals_TTMS = pd.Series(TTMS_Signals_TTMS, index=df.index)

    basicLongCondition_TTMS = (TTMS_Signals_TTMS == 1) if redGreen_TTMS else (TTMS_Signals_TTMS > 0)
    basicShortCondition_TTMS = (TTMS_Signals_TTMS == -1) if redGreen_TTMS else (TTMS_Signals_TTMS < 0)

    TTMS_SignalsLong_TTMS = basicLongCondition_TTMS if not highlightMovements_TTMS else (NoSqz_TTMS & basicLongCondition_TTMS)
    TTMS_SignalsShort_TTMS = basicShortCondition_TTMS if not highlightMovements_TTMS else (NoSqz_TTMS & basicShortCondition_TTMS)

    TTMS_SignalsLongCross_TTMS = TTMS_SignalsLong_TTMS if not cross_TTMS else (~TTMS_SignalsLong_TTMS.shift(1).fillna(False) & TTMS_SignalsLong_TTMS)
    TTMS_SignalsShortCross_TTMS = TTMS_SignalsShort_TTMS if not cross_TTMS else (~TTMS_SignalsShort_TTMS.shift(1).fillna(False) & TTMS_SignalsShort_TTMS)

    TTMS_SignalsLongFinal_TTMS = TTMS_SignalsLongCross_TTMS if not inverse_TTMS else TTMS_SignalsShortCross_TTMS
    TTMS_SignalsShortFinal_TTMS = TTMS_SignalsShortCross_TTMS if not inverse_TTMS else TTMS_SignalsLongCross_TTMS

    if not use_TTMS:
        TTMS_SignalsLongFinal_TTMS = pd.Series(True, index=df.index)
        TTMS_SignalsShortFinal_TTMS = pd.Series(True, index=df.index)

    if inverse_TTMS:
        TTMS_SignalsLongFinal_TTMS_orig = TTMS_SignalsLongFinal_TTMS.copy()
        TTMS_SignalsLongFinal_TTMS = TTMS_SignalsShortCross_TTMS
        TTMS_SignalsShortFinal_TTMS = TTMS_SignalsLongFinal_TTMS_orig

    if not useE2PSS:
        signalLongE2PSSFinal = pd.Series(True, index=df.index)
        signalShortE2PSSFinal = pd.Series(True, index=df.index)

    # ─────────────────────────────────────────────────────────────────────────────
    # Generate Entries
    # ─────────────────────────────────────────────────────────────────────────────
    entries_list = []
    trade_num = 1

    long_entry_cond = signalLongE2PSSFinal & TTMS_SignalsLongFinal_TTMS
    short_entry_cond = signalShortE2PSSFinal & TTMS_SignalsShortFinal_TTMS

    for i in range(len(df)):
        if i < 2:
            continue

        close_i = df['close'].iloc[i]
        open_i = df['open'].iloc[i]
        high_i = df['high'].iloc[i]
        low_i = df['low'].iloc[i]

        if long_entry_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = close_i
            entries_list.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

        if short_entry_cond.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_price = close_i
            entries_list.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries_list