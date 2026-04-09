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
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    devKC_TTMS = tr.rolling(length_TTMS).mean()
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    KC_upper_high_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_high_TTMS
    KC_lower_high_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_high_TTMS
    KC_upper_mid_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_mid_TTMS
    KC_lower_mid_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_mid_TTMS
    
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    LowSqz_TTMS = (BB_lower_TTMS >= KC_lower_low_TTMS) | (BB_upper_TTMS <= KC_upper_low_TTMS)
    MidSqz_TTMS = (BB_lower_TTMS >= KC_lower_mid_TTMS) | (BB_upper_TTMS <= KC_upper_mid_TTMS)
    HighSqz_TTMS = (BB_lower_TTMS >= KC_lower_high_TTMS) | (BB_upper_TTMS <= KC_upper_high_TTMS)
    
    lengthT3 = 5
    factorT3 = 0.7
    srcT3 = close
    
    def gdT3(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + factorT3) - ema2 * factorT3
    
    t3 = gdT3(gdT3(gdT3(srcT3, lengthT3), lengthT3), lengthT3)
    
    t3Signals = np.where(t3 > t3.shift(1), 1, -1)
    basicLongCondition = (t3Signals > 0) & (close > t3)
    basicShortCondition = (t3Signals < 0) & (close < t3)
    
    t3SignalsLong = basicLongCondition
    t3SignalsShort = basicShortCondition
    
    t3SignalsLongCross = (~t3SignalsLong.shift(1).fillna(False)) & t3SignalsLong
    t3SignalsShortCross = (~t3SignalsShort.shift(1).fillna(False)) & t3SignalsShort
    
    t3SignalsLongFinal = t3SignalsLongCross
    t3SignalsShortFinal = t3SignalsShortCross
    
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    avg_center = (highest_high + lowest_low + close.rolling(length_TTMS).mean()) / 3
    mom_TTMS_raw = close - avg_center
    
    def linreg(series, length):
        x = np.arange(length)
        x_mean = (length - 1) / 2.0
        result = pd.Series(index=series.index, dtype=float)
        for i in range(length - 1, len(series)):
            y = series.iloc[i - length + 1:i + 1].values
            y_mean = np.mean(y)
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            if denominator != 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean
                result.iloc[i] = slope * (length - 1 - x_mean) + intercept
            else:
                result.iloc[i] = 0
        return result
    
    mom_TTMS = linreg(mom_TTMS_raw, length_TTMS)
    
    iff_1_TTMS_no = np.where(mom_TTMS > mom_TTMS.shift(1).fillna(0), 1, 2)
    iff_2_TTMS_no = np.where(mom_TTMS < mom_TTMS.shift(1).fillna(0), -1, -2)
    TTMS_Signals_TTMS = np.where(mom_TTMS > 0, iff_1_TTMS_no, iff_2_TTMS_no)
    
    basicLongCondition_TTMS = (TTMS_Signals_TTMS == 1)
    basicShortCondition_TTMS = (TTMS_Signals_TTMS == -1)
    
    TTMS_SignalsLong_TTMS = NoSqz_TTMS & basicLongCondition_TTMS
    TTMS_SignalsShort_TTMS = NoSqz_TTMS & basicShortCondition_TTMS
    
    TTMS_SignalsLongCross_TTMS = (~TTMS_SignalsLong_TTMS.shift(1).fillna(False)) & TTMS_SignalsLong_TTMS
    TTMS_SignalsShortCross_TTMS = (~TTMS_SignalsShort_TTMS.shift(1).fillna(False)) & TTMS_SignalsShort_TTMS
    
    TTMS_SignalsLongFinal_TTMS = TTMS_SignalsLongCross_TTMS
    TTMS_SignalsShortFinal_TTMS = TTMS_SignalsShortCross_TTMS
    
    entry_condition = TTMS_SignalsShortFinal_TTMS & (close < t3)
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if entry_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
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
    
    return entries