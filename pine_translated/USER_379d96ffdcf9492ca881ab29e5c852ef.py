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
    open_col = df['open']
    volume = df['volume']
    time_col = df['time']

    length_TTMS = 20
    BB_mult_TTMS = 2.0
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = BB_mult_TTMS * close.rolling(length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS

    KC_mult_high_TTMS = 1.0
    KC_mult_mid_TTMS = 1.5
    KC_mult_low_TTMS = 2.0
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
    sma_close = close.rolling(length_TTMS).mean()
    momentum_base = close - ((highest_high + lowest_low) / 2 + sma_close) / 2
    mom_TTMS = momentum_base.rolling(length_TTMS).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] * (len(x) - 1) / 2 + np.mean(x) if len(x) > 0 else np.nan, raw=True)

    mom_prev = mom_TTMS.shift(1).fillna(0)
    iff_1_TTMS_no = np.where(mom_TTMS > mom_prev, 1, 2)
    iff_2_TTMS_no = np.where(mom_TTMS < mom_prev, -1, -2)
    TTMS_Signals_TTMS = np.where(mom_TTMS > 0, iff_1_TTMS_no, iff_2_TTMS_no)

    basicLongCondition_TTMS = np.where(TTMS_Signals_TTMS == 1, True, False)
    basicShortCondition_TTMS = np.where(TTMS_Signals_TTMS == -1, True, False)

    TTMS_SignalsLong_TTMS = basicLongCondition_TTMS & NoSqz_TTMS
    TTMS_SignalsShort_TTMS = basicShortCondition_TTMS & NoSqz_TTMS

    TTMS_SignalsLongPrev_TTMS = TTMS_SignalsLong_TTMS.shift(1).fillna(False)
    TTMS_SignalsShortPrev_TTMS = TTMS_SignalsShort_TTMS.shift(1).fillna(False)
    TTMS_SignalsLongCross_TTMS = (~TTMS_SignalsLongPrev_TTMS) & TTMS_SignalsLong_TTMS
    TTMS_SignalsShortCross_TTMS = (~TTMS_SignalsShortPrev_TTMS) & TTMS_SignalsShort_TTMS

    TTMS_SignalsLongFinal_TTMS = TTMS_SignalsLongCross_TTMS
    TTMS_SignalsShortFinal_TTMS = TTMS_SignalsShortCross_TTMS

    lengthT3 = 5
    factorT3 = 0.7
    srcT3 = close

    def gdT3_series(series, length):
        ema1 = series.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factorT3) - ema2 * factorT3

    t3 = gdT3_series(gdT3_series(gdT3_series(srcT3, lengthT3), lengthT3), lengthT3)
    t3_prev = t3.shift(1)
    t3Signals = np.where(t3 > t3_prev, 1, -1)

    basicLongCondition = (t3Signals > 0) & (close > t3)
    basicShortCondition = (t3Signals < 0) & (close < t3)

    t3SignalsLong = basicLongCondition
    t3SignalsShort = basicShortCondition

    t3SignalsLongPrev = t3SignalsLong.shift(1).fillna(False)
    t3SignalsShortPrev = t3SignalsShort.shift(1).fillna(False)
    t3SignalsLongCross = (~t3SignalsLongPrev) & t3SignalsLong
    t3SignalsShortCross = (~t3SignalsShortPrev) & t3SignalsShort

    t3SignalsLongFinal = t3SignalsLongCross
    t3SignalsShortFinal = t3SignalsShortCross

    long_condition = TTMS_SignalsLongFinal_TTMS & t3SignalsLongFinal
    short_condition = TTMS_SignalsShortFinal_TTMS & t3SignalsShortFinal

    entries = []
    trade_num = 1
    position_open = False

    for i in range(len(df)):
        if pd.isna(mom_TTMS.iloc[i]) or pd.isna(t3.iloc[i]):
            continue

        if position_open:
            if long_condition.iloc[i]:
                position_open = False
            elif short_condition.iloc[i]:
                position_open = False

        if not position_open:
            if long_condition.iloc[i]:
                ts = int(time_col.iloc[i])
                entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                entry_price = float(close.iloc[i])

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
                position_open = True
            elif short_condition.iloc[i]:
                ts = int(time_col.iloc[i])
                entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                entry_price = float(close.iloc[i])

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
                position_open = True

    return entries