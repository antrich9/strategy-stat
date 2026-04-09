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
    
    # Default input values from Pine Script
    useT3 = True
    factorT3 = 0.7
    signalTypeT3 = 'MA + Price'
    inverseT3 = False
    crossOnlyT3 = True
    
    # T3 calculation
    def calc_t3(src, length):
        e1 = src.ewm(span=length, adjust=False).mean()
        e2 = e1.ewm(span=length, adjust=False).mean()
        e3 = e2.ewm(span=length, adjust=False).mean()
        e4 = e3.ewm(span=length, adjust=False).mean()
        e5 = e4.ewm(span=length, adjust=False).mean()
        e6 = e5.ewm(span=length, adjust=False).mean()
        c1 = -factorT3 ** 3
        c2 = 3 * factorT3 ** 2 + 3 * factorT3 ** 3
        c3 = -6 * factorT3 ** 2 - 3 * factorT3 - 3 * factorT3 ** 3
        c4 = 1 + 3 * factorT3 + factorT3 ** 3 + 3 * factorT3 ** 2
        return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    
    t3_25 = calc_t3(close, 25)
    t3_100 = calc_t3(close, 100)
    t3_200 = calc_t3(close, 200)
    
    # T3 indicator conditions
    longConditionIndiT3 = (close > t3_25) & (close > t3_100) & (close > t3_200)
    shortConditionIndiT3 = (close < t3_25) & (close < t3_100) & (close < t3_200)
    longConditionIndiT3MA = (t3_100 < t3_25) & (t3_200 < t3_100)
    shortConditionIndiT3MA = (t3_100 > t3_25) & (t3_200 > t3_100)
    
    # Signal entry logic for T3
    if signalTypeT3 == 'MA + Price':
        signalEntryLongT3 = longConditionIndiT3 & longConditionIndiT3MA
        signalEntryShortT3 = shortConditionIndiT3 & shortConditionIndiT3MA
    elif signalTypeT3 == 'MA Only':
        signalEntryLongT3 = longConditionIndiT3MA
        signalEntryShortT3 = shortConditionIndiT3MA
    else:
        signalEntryLongT3 = longConditionIndiT3
        signalEntryShortT3 = shortConditionIndiT3
    
    # Apply cross only filter
    if crossOnlyT3:
        signalEntryLongT3_prev = signalEntryLongT3.shift(1).fillna(False).astype(bool)
        signalEntryShortT3_prev = signalEntryShortT3.shift(1).fillna(False).astype(bool)
        signalEntryLongT3 = signalEntryLongT3.astype(bool) & ~signalEntryLongT3_prev
        signalEntryShortT3 = signalEntryShortT3.astype(bool) & ~signalEntryShortT3_prev
    
    # Apply inverse
    if inverseT3:
        finalLongSignalT3 = signalEntryShortT3
        finalShortSignalT3 = signalEntryLongT3
    else:
        finalLongSignalT3 = signalEntryLongT3
        finalShortSignalT3 = signalEntryShortT3
    
    # Trendilo calculation
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    pct_change = close.pct_change(trendilo_smooth) * 100
    
    # ALMA implementation
    def alma(series, length, offset, sigma):
        m = np.floor(offset * (length - 1))
        s = sigma / length
        w = np.exp(-((np.arange(length) - m) ** 2) / (2 * s * s))
        w = w / w.sum()
        return pd.Series(np.convolve(series, w, mode='valid'), index=series.index[length-1:])
    
    avg_pct_change = alma(pct_change.values, trendilo_length, trendilo_offset, trendilo_sigma)
    avg_pct_change = avg_pct_change.reindex(close.index).ffill().bfill()
    
    rms_vals = []
    for i in range(trendilo_length - 1, len(close)):
        window = avg_pct_change.iloc[i - trendilo_length + 1:i + 1]
        rms = trendilo_bmult * np.sqrt((window ** 2).mean())
        rms_vals.append(rms)
    
    rms_series = pd.Series([np.nan] * (trendilo_length - 1) + rms_vals, index=close.index)
    
    trendilo_dir = pd.Series(0, index=close.index)
    trendilo_dir[avg_pct_change > rms_series] = 1
    trendilo_dir[avg_pct_change < -rms_series] = -1
    
    # TTM Squeeze calculation
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = BB_mult_TTMS * close.rolling(length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS
    
    KC_mult_low_TTMS = 2.0
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    devKC_TTMS = tr.rolling(length_TTMS).mean()
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    
    # Momentum calculation
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    sma_close = close.rolling(length_TTMS).mean()
    avg_val = (highest_high + lowest_low) / 2 + sma_close
    mom_TTMS = (close - avg_val).rolling(length_TTMS).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] * (len(x) - 1) + np.polyfit(np.arange(len(x)), x, 1)[1], raw=True)
    
    mom_prev = mom_TTMS.shift(1).fillna(0)
    TTMS_Signals_TTMS = pd.Series(0, index=close.index)
    TTMS_Signals_TTMS[mom_TTMS > 0] = np.where(mom_TTMS[mom_TTMS > 0] > mom_prev[mom_TTMS > 0], 1, 2)
    TTMS_Signals_TTMS[mom_TTMS <= 0] = np.where(mom_TTMS[mom_TTMS <= 0] < mom_prev[mom_TTMS <= 0], -1, -2)
    
    basicLongCondition_TTMS = TTMS_Signals_TTMS == 1
    basicShortCondition_TTMS = TTMS_Signals_TTMS == -1
    
    highlightMovements_TTMS = True
    TTMS_SignalsLong_TTMS = NoSqz_TTMS & basicLongCondition_TTMS
    TTMS_SignalsShort_TTMS = NoSqz_TTMS & basicShortCondition_TTMS
    
    cross_TTMS = True
    if cross_TTMS:
        TTMS_SignalsLongPrev = TTMS_SignalsLong_TTMS.shift(1).fillna(False).astype(bool)
        TTMS_SignalsShortPrev = TTMS_SignalsShort_TTMS.shift(1).fillna(False).astype(bool)
        TTMS_SignalsLongCross_TTMS = TTMS_SignalsLong_TTMS.astype(bool) & ~TTMS_SignalsLongPrev
        TTMS_SignalsShortCross_TTMS = TTMS_SignalsShort_TTMS.astype(bool) & ~TTMS_SignalsShortPrev
    else:
        TTMS_SignalsLongCross_TTMS = TTMS_SignalsLong_TTMS
        TTMS_SignalsShortCross_TTMS = TTMS_SignalsShort_TTMS
    
    use_TTMS = True
    inverse_TTMS = False
    if use_TTMS:
        if inverse_TTMS:
            TTMS_SignalsLongFinal_TTMS = TTMS_SignalsShortCross_TTMS
            TTMS_SignalsShortFinal_TTMS = TTMS_SignalsLongCross_TTMS
        else:
            TTMS_SignalsLongFinal_TTMS = TTMS_SignalsLongCross_TTMS
            TTMS_SignalsShortFinal_TTMS = TTMS_SignalsShortCross_TTMS
    else:
        TTMS_SignalsLongFinal_TTMS = pd.Series(True, index=close.index)
        TTMS_SignalsShortFinal_TTMS = pd.Series(True, index=close.index)
    
    # Final entry conditions
    long_condition = finalLongSignalT3.astype(bool) & (trendilo_dir == 1) & basicLongCondition_TTMS.astype(bool)
    short_condition = finalShortSignalT3.astype(bool) & (trendilo_dir == -1) & basicShortCondition_TTMS.astype(bool)
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if np.isnan(t3_25.iloc[i]) or np.isnan(t3_100.iloc[i]) or np.isnan(t3_200.iloc[i]):
            continue
        if np.isnan(mom_TTMS.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            entry_price = close.iloc[i]
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
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
        
        if short_condition.iloc[i]:
            entry_price = close.iloc[i]
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
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