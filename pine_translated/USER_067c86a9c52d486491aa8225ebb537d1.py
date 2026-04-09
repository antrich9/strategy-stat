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
    n = len(df)

    # E2PSS Parameters
    PeriodE2PSS = 15
    pi = 2 * np.pi

    a1 = np.exp(-1.414 * pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3

    Filt2 = np.zeros(n)
    TriggerE2PSS = np.zeros(n)

    for i in range(n):
        if i < 2:
            Filt2[i] = close.iloc[i]
        else:
            Filt2[i] = coef1 * close.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
        TriggerE2PSS[i] = Filt2[i-1] if i > 0 else 0

    signalLongE2PSS = Filt2 > TriggerE2PSS
    signalShortE2PSS = Filt2 < TriggerE2PSS

    signalLongE2PSSFinal = signalLongE2PSS
    signalShortE2PSSFinal = signalShortE2PSS

    # Trendilo Parameters
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0

    pct_change = close.diff(trendilo_smooth) / close * 100

    def alma(arr, length, offset, sigma):
        window = np.arange(length)
        m = offset * (length - 1)
        s = sigma * (length - 1) / 6
        w = np.exp(-((window - m) ** 2) / (2 * s * s))
        w = w / w.sum()
        result = np.convolve(arr, w, mode='same')
        return result

    avg_pct_change = alma(pct_change.values, trendilo_length, trendilo_offset, trendilo_sigma)
    avg_pct_change_series = pd.Series(avg_pct_change, index=pct_change.index)

    rms_vals = np.zeros(n)
    for i in range(n):
        if i >= trendilo_length - 1:
            start_idx = i - trendilo_length + 1
            rms_vals[i] = trendilo_bmult * np.sqrt(np.mean(avg_pct_change[start_idx:i+1] ** 2))
        else:
            rms_vals[i] = 0

    rms = pd.Series(rms_vals, index=pct_change.index)
    trendilo_dir = pd.Series(0, index=pct_change.index)
    trendilo_dir[avg_pct_change_series > rms] = 1
    trendilo_dir[avg_pct_change_series < -rms] = -1

    # TTM Squeeze Parameters
    length_TTMS = 20
    BB_mult_TTMS = 2.0

    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = close.rolling(length_TTMS).std() * BB_mult_TTMS
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS

    KC_mult_high_TTMS = 1.0
    KC_mult_mid_TTMS = 1.5
    KC_mult_low_TTMS = 2.0
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
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

    # Momentum
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    mom_TTMS_raw = close - (highest_high + lowest_low) / 2
    mom_TTMS = mom_TTMS_raw.rolling(length_TTMS).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] * (len(x) - 1) / 2 + x.mean(), raw=True)

    iff_1_TTMS_no = pd.Series(1, index=mom_TTMS.index)
    iff_1_TTMS_no[mom_TTMS <= mom_TTMS.shift(1)] = 2
    iff_2_TTMS_no = pd.Series(-1, index=mom_TTMS.index)
    iff_2_TTMS_no[mom_TTMS >= mom_TTMS.shift(1)] = -2

    TTMS_Signals_TTMS = pd.Series(0, index=mom_TTMS.index)
    TTMS_Signals_TTMS[mom_TTMS > 0] = iff_1_TTMS_no[mom_TTMS > 0]
    TTMS_Signals_TTMS[mom_TTMS < 0] = iff_2_TTMS_no[mom_TTMS < 0]

    redGreen_TTMS = True
    basicLongCondition_TTMS = TTMS_Signals_TTMS == 1
    basicShortCondition_TTMS = TTMS_Signals_TTMS == -1

    highlightMovements_TTMS = True
    TTMS_SignalsLong_TTMS = NoSqz_TTMS & basicLongCondition_TTMS
    TTMS_SignalsShort_TTMS = NoSqz_TTMS & basicShortCondition_TTMS

    cross_TTMS = True
    TTMS_SignalsLongCross_TTMS = (~TTMS_SignalsLong_TTMS.shift(1).fillna(False)) & TTMS_SignalsLong_TTMS
    TTMS_SignalsShortCross_TTMS = (~TTMS_SignalsShort_TTMS.shift(1).fillna(False)) & TTMS_SignalsShort_TTMS

    use_TTMS = True
    inverse_TTMS = False
    TTMS_SignalsLongFinal_TTMS = TTMS_SignalsLongCross_TTMS
    TTMS_SignalsShortFinal_TTMS = TTMS_SignalsShortCross_TTMS

    # London Time Windows (approximated - exact timestamps require timezone info)
    # Using hour extraction assuming time is in UTC
    time_series = pd.to_datetime(df['time'], unit='s', utc=True)
    hour = time_series.dt.hour
    minute = time_series.dt.minute

    isWithinMorningWindow = ((hour == 7) & (minute >= 45)) | ((hour >= 8) & (hour < 11)) | ((hour == 11) & (minute < 45))
    isWithinAfternoonWindow = ((hour >= 14) & (hour < 16)) | ((hour == 16) & (minute < 45))

    in_trading_window = isWithinMorningWindow

    # HTF EMA and trend
    htf_tf = "240"
    htf_ema = close.ewm(span=50, adjust=False).mean()

    htf_trend_up = (close > htf_ema) & (htf_ema > htf_ema.shift(1))
    htf_trend_dn = (close < htf_ema) & (htf_ema < htf_ema.shift(1))

    # Entry Conditions
    long_condition = signalLongE2PSSFinal & (trendilo_dir == 1) & basicLongCondition_TTMS & in_trading_window & htf_trend_up
    short_condition = signalShortE2PSSFinal & (trendilo_dir == -1) & basicShortCondition_TTMS & in_trading_window & htf_trend_dn

    entries = []
    trade_num = 1

    for i in range(1, n):
        if pd.isna(Filt2[i]) or pd.isna(avg_pct_change_series.iloc[i]) or pd.isna(mom_TTMS.iloc[i]):
            continue

        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

        if short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

    return entries