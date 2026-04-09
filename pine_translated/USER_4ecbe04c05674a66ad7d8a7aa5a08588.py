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
    close = df['close'].copy()
    high = df['high'].copy()
    low = df['low'].copy()
    open_price = df['open'].copy()
    time_col = df['time'].copy()

    # Inputs
    useE2PSS = True
    inverseE2PSS = False
    PeriodE2PSS = 15

    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0

    length_TTMS = 20
    BB_mult_TTMS = 2.0

    use_TTMS = True
    redGreen_TTMS = True
    cross_TTMS = True
    inverse_TTMS = False
    highlightMovements_TTMS = True

    atrLength = 14

    # E2PSS Filter Implementation
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * np.pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / PeriodE2PSS)
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

    signalLongE2PSS = Filt2 > TriggerE2PSS if useE2PSS else np.ones(len(df), dtype=bool)
    signalShortE2PSS = Filt2 < TriggerE2PSS if useE2PSS else np.ones(len(df), dtype=bool)

    if inverseE2PSS:
        signalLongE2PSSFinal = signalShortE2PSS.copy()
        signalShortE2PSSFinal = signalLongE2PSS.copy()
    else:
        signalLongE2PSSFinal = signalLongE2PSS.copy()
        signalShortE2PSSFinal = signalShortE2PSS.copy()

    # Trendilo Implementation
    pct_change = close.diff(trendilo_smooth) / close * 100

    def alma(arr, length, offset, sigma):
        w = np.exp(-np.square(np.arange(length) - offset * (length - 1)) / (2 * sigma * sigma))
        w = w / w.sum()
        result = np.convolve(arr, w, mode='valid')
        pad = np.full(length - 1, np.nan)
        return np.concatenate([pad, result])

    avg_pct_change = alma(pct_change.values, trendilo_length, trendilo_offset, trendilo_sigma)
    avg_pct_change = pd.Series(avg_pct_change, index=df.index)

    rms_values = np.full(len(df), np.nan)
    for i in range(trendilo_length - 1, len(df)):
        window = avg_pct_change.iloc[i - trendilo_length + 1:i + 1]
        rms_values[i] = trendilo_bmult * np.sqrt(np.mean(window * window))

    trendilo_dir = np.where(avg_pct_change > rms_values, 1,
                            np.where(avg_pct_change < -rms_values, -1, 0))
    trendilo_dir = pd.Series(trendilo_dir, index=df.index)

    # TTMS Implementation - Bollinger Bands
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    BB_std = close.rolling(length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + BB_mult_TTMS * BB_std
    BB_lower_TTMS = BB_basis_TTMS - BB_mult_TTMS * BB_std

    # Keltner Channels
    KC_basis_TTMS = close.rolling(length_TTMS).mean()

    def wilder_tr(series):
        tr = np.zeros(len(series))
        tr[0] = series.iloc[0]
        for i in range(1, len(series)):
            high_low = high.iloc[i] - low.iloc[i]
            high_close = abs(high.iloc[i] - close.iloc[i-1])
            low_close = abs(low.iloc[i] - close.iloc[i-1])
            tr[i] = max(high_low, high_close, low_close)
        return pd.Series(tr, index=series.index)

    tr_series = wilder_tr(low)
    devKC_TTMS = tr_series.ewm(alpha=1.0/length_TTMS, adjust=False).mean()

    KC_upper_high_TTMS = KC_basis_TTMS + devKC_TTMS * 1.0
    KC_lower_high_TTMS = KC_basis_TTMS - devKC_TTMS * 1.0
    KC_upper_mid_TTMS = KC_basis_TTMS + devKC_TTMS * 1.5
    KC_lower_mid_TTMS = KC_basis_TTMS - devKC_TTMS * 1.5
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * 2.0
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * 2.0

    # Squeeze Conditions
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    LowSqz_TTMS = (BB_lower_TTMS >= KC_lower_low_TTMS) | (BB_upper_TTMS <= KC_upper_low_TTMS)
    MidSqz_TTMS = (BB_lower_TTMS >= KC_lower_mid_TTMS) | (BB_upper_TTMS <= KC_upper_mid_TTMS)
    HighSqz_TTMS = (BB_lower_TTMS >= KC_lower_high_TTMS) | (BB_upper_TTMS <= KC_upper_high_TTMS)

    # Momentum Oscillator (Linear Regression)
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    sma_close = close.rolling(length_TTMS).mean()
    avg_value = (highest_high + lowest_low) / 2
    mom_input = close - (avg_value + sma_close) / 2

    def linreg(series, length, offset=0):
        result = np.full(len(series), np.nan)
        for i in range(length - 1, len(series)):
            y = series.iloc[i - length + 1:i + 1].values
            x = np.arange(length)
            x_mean = (length - 1) / 2.0
            y_mean = np.mean(y)
            slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
            intercept = y_mean - slope * x_mean
            result[i] = slope * (length - 1 - offset) + intercept
        return pd.Series(result, index=series.index)

    mom_TTMS = linreg(mom_input, length_TTMS, 0)

    # Momentum color and signals
    mom_prev = mom_TTMS.shift(1).fillna(0)
    iff_1_TTMS_no = np.where(mom_TTMS > mom_prev, 1, 2)
    iff_2_TTMS_no = np.where(mom_TTMS < mom_prev, -1, -2)

    TTMS_Signals_TTMS = np.where(mom_TTMS > 0, iff_1_TTMS_no, iff_2_TTMS_no)
    TTMS_Signals_TTMS = pd.Series(TTMS_Signals_TTMS, index=df.index)

    basicLongCondition_TTMS = (TTMS_Signals_TTMS == 1) if redGreen_TTMS else (TTMS_Signals_TTMS > 0)
    basicShortCondition_TTMS = (TTMS_Signals_TTMS == -1) if redGreen_TTMS else (TTMS_Signals_TTMS < 0)

    TTMS_SignalsLong_TTMS = basicLongCondition_TTMS if not highlightMovements_TTMS else (NoSqz_TTMS & basicLongCondition_TTMS)
    TTMS_SignalsShort_TTMS = basicShortCondition_TTMS if not highlightMovements_TTMS else (NoSqz_TTMS & basicShortCondition_TTMS)

    TTMS_SignalsLongPrev = TTMS_SignalsLong_TTMS.shift(1).fillna(False).astype(bool)
    TTMS_SignalsShortPrev = TTMS_SignalsShort_TTMS.shift(1).fillna(False).astype(bool)

    TTMS_SignalsLongCross_TTMS = (~TTMS_SignalsLongPrev & TTMS_SignalsLong_TTMS) if cross_TTMS else TTMS_SignalsLong_TTMS
    TTMS_SignalsShortCross_TTMS = (~TTMS_SignalsShortPrev & TTMS_SignalsShort_TTMS) if cross_TTMS else TTMS_SignalsShort_TTMS

    TTMS_SignalsLongFinal_TTMS = TTMS_SignalsShortCross_TTMS if (use_TTMS and inverse_TTMS) else TTMS_SignalsLongCross_TTMS
    TTMS_SignalsShortFinal_TTMS = TTMS_SignalsLongCross_TTMS if (use_TTMS and inverse_TTMS) else TTMS_SignalsShortCross_TTMS

    # ATR calculation using Wilder's method
    def wilder_atr(high_series, low_series, close_series, length):
        tr = np.zeros(len(high_series))
        tr[0] = high_series.iloc[0] - low_series.iloc[0]
        for i in range(1, len(high_series)):
            high_low = high_series.iloc[i] - low_series.iloc[i]
            high_close = abs(high_series.iloc[i] - close_series.iloc[i-1])
            low_close = abs(low_series.iloc[i] - close_series.iloc[i-1])
            tr[i] = max(high_low, high_close, low_close)
        atr = np.zeros(len(high_series))
        atr[length-1] = np.mean(tr[:length])
        for i in range(length, len(high_series)):
            atr[i] = (atr[i-1] * (length - 1) + tr[i]) / length
        return pd.Series(atr, index=high_series.index)

    atr1 = wilder_atr(high, low, close, atrLength)

    # Weekend filter
    dt = pd.to_datetime(df['time'], unit='s', utc=True)
    day_of_week = dt.dayofweek
    hour = dt.hour

    notWeekend = (day_of_week != 4) | ((day_of_week == 4) & (hour < 16))
    notWeekend = notWeekend & ((day_of_week != 5) & ((day_of_week != 6) | (hour >= 22)))

    # Entry conditions
    long_condition = signalLongE2PSSFinal & (trendilo_dir == 1) & basicLongCondition_TTMS
    short_condition = signalShortE2PSSFinal & (trendilo_dir == -1) & basicShortCondition_TTMS

    long_entry_cond = long_condition & notWeekend
    short_entry_cond = short_condition & notWeekend

    # Generate entries
    entries = []
    trade_num = 1
    position_open = False

    for i in range(len(df)):
        if position_open:
            continue

        if long_entry_cond.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = int(time_col.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()

            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            position_open = True

        elif short_entry_cond.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = int(time_col.iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()

            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            position_open = True

    return entries