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
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"df must contain columns: {required_cols}")

    close = df['close']
    high = df['high']
    low = df['low']

    # T3 Parameters
    factorT3 = 0.7

    # T3 calculation
    def calc_t3(src, length):
        e1 = src.ewm(span=length, adjust=False).mean()
        e2 = e1.ewm(span=length, adjust=False).mean()
        e3 = e2.ewm(span=length, adjust=False).mean()
        e4 = e3.ewm(span=length, adjust=False).mean()
        e5 = e4.ewm(span=length, adjust=False).mean()
        e6 = e5.ewm(span=length, adjust=False).mean()
        c1 = -factorT3 * factorT3 * factorT3
        c2 = 3 * factorT3 * factorT3 + 3 * factorT3 * factorT3 * factorT3
        c3 = -6 * factorT3 * factorT3 - 3 * factorT3 - 3 * factorT3 * factorT3 * factorT3
        c4 = 1 + 3 * factorT3 + factorT3 * factorT3 * factorT3 + 3 * factorT3 * factorT3
        return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    t3_25 = calc_t3(close, 25)
    t3_100 = calc_t3(close, 100)
    t3_200 = calc_t3(close, 200)

    # T3 conditions
    longConditionIndiT3 = (close > t3_25) & (close > t3_100) & (close > t3_200)
    shortConditionIndiT3 = (close < t3_25) & (close < t3_100) & (close < t3_200)
    longConditionIndiT3MA = (t3_100 < t3_25) & (t3_200 < t3_100)
    shortConditionIndiT3MA = (t3_100 > t3_25) & (t3_200 > t3_100)

    # Signal type 'MA + Price' (default), useT3=true, crossOnlyT3=true, inverseT3=false
    signalEntryLongT3 = longConditionIndiT3 & longConditionIndiT3MA
    signalEntryShortT3 = shortConditionIndiT3 & shortConditionIndiT3MA

    finalLongSignalT3 = signalEntryLongT3 & (~signalEntryLongT3.shift(1).fillna(False))
    finalShortSignalT3 = signalEntryShortT3 & (~signalEntryShortT3.shift(1).fillna(False))

    # Trendilo
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0

    pct_change = close.diff(trendilo_smooth) / close * 100
    avg_pct_change = pct_change.ewm(span=trendilo_length, adjust=False).mean()
    rms = trendilo_bmult * np.sqrt((avg_pct_change ** 2).rolling(trendilo_length).mean())
    trendilo_dir = np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0))
    trendilo_dir = pd.Series(trendilo_dir, index=df.index)

    # TTMS - TTM Squeeze
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    KC_mult_high_TTMS = 1.0
    KC_mult_mid_TTMS = 1.5
    KC_mult_low_TTMS = 2.0

    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = BB_mult_TTMS * close.rolling(length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS

    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
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

    # Momentum (linreg approximation using rolling covariance)
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    avg_price = (highest_high + lowest_low) / 2
    mom_TTMS = close - avg_price.ewm(span=length_TTMS, adjust=False).mean()
    mom_TTMS = mom_TTMS.rolling(length_TTMS).mean()

    # TTMS Signals
    mom_prev = mom_TTMS.shift(1).fillna(0)
    TTMS_Signals_TTMS = pd.Series(np.where(mom_TTMS > 0, np.where(mom_TTMS > mom_prev, 1, 2), np.where(mom_TTMS < mom_prev, -1, -2)), index=df.index)

    basicLongCondition_TTMS = TTMS_Signals_TTMS > 0
    basicShortCondition_TTMS = TTMS_Signals_TTMS < 0

    TTMS_SignalsLong_TTMS = NoSqz_TTMS & basicLongCondition_TTMS
    TTMS_SignalsShort_TTMS = NoSqz_TTMS & basicShortCondition_TTMS

    # Cross conditions
    TTMS_SignalsLongCross_TTMS = (~TTMS_SignalsLong_TTMS.shift(1).fillna(False)) & TTMS_SignalsLong_TTMS
    TTMS_SignalsShortCross_TTMS = (~TTMS_SignalsShort_TTMS.shift(1).fillna(False)) & TTMS_SignalsShort_TTMS

    use_TTMS = True
    inverse_TTMS = False

    TTMS_SignalsLongFinal_TTMS = TTMS_SignalsLongCross_TTMS if use_TTMS and not inverse_TTMS else TTMS_SignalsShortCross_TTMS if use_TTMS else True
    TTMS_SignalsShortFinal_TTMS = TTMS_SignalsShortCross_TTMS if use_TTMS and not inverse_TTMS else TTMS_SignalsLongCross_TTMS if use_TTMS else True

    # Entry conditions
    long_condition = finalLongSignalT3 & (trendilo_dir == 1) & basicLongCondition_TTMS
    short_condition = finalShortSignalT3 & (trendilo_dir == -1) & basicShortCondition_TTMS

    # Generate entries
    entries = []
    trade_num = 1

    for i in range(1, len(df)):
        if pd.isna(t3_25.iloc[i]) or pd.isna(t3_100.iloc[i]) or pd.isna(t3_200.iloc[i]):
            continue
        if pd.isna(trendilo_dir.iloc[i]):
            continue
        if pd.isna(mom_TTMS.iloc[i]):
            continue

        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

        if short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries