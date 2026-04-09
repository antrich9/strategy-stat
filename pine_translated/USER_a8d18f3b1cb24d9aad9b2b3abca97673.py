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

    # === Triple T3 Parameters ===
    factorT3 = 0.7
    srcT3 = df['close']

    # T3 calculation function
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

    t3_25 = calc_t3(srcT3, 25)
    t3_100 = calc_t3(srcT3, 100)
    t3_200 = calc_t3(srcT3, 200)

    # T3 conditions (signalTypeT3 defaults to 'MA + Price', crossOnlyT3 defaults to true)
    longConditionIndiT3 = (df['close'] > t3_25) & (df['close'] > t3_100) & (df['close'] > t3_200)
    shortConditionIndiT3 = (df['close'] < t3_25) & (df['close'] < t3_100) & (df['close'] < t3_200)
    longConditionIndiT3MA = (t3_100 < t3_25) & (t3_200 < t3_100)
    shortConditionIndiT3MA = (t3_100 > t3_25) & (t3_200 > t3_100)

    # Combined T3 entry signals with crossOnly logic
    signalEntryLongT3 = longConditionIndiT3 & longConditionIndiT3MA
    signalEntryShortT3 = shortConditionIndiT3 & shortConditionIndiT3MA

    # Cross only logic: signal is true only on transition from false to true
    finalLongSignalT3 = signalEntryLongT3 & ~signalEntryLongT3.shift(1).fillna(False)
    finalShortSignalT3 = signalEntryShortT3 & ~signalEntryShortT3.shift(1).fillna(False)

    # === TDFI Parameters ===
    lookbackTDFI = 13
    mmaLengthTDFI = 13
    mmaModeTDFI = 'ema'
    smmaLengthTDFI = 13
    smmaModeTDFI = 'ema'
    nLengthTDFI = 3
    filterHighTDFI = 0.05
    filterLowTDFI = -0.05

    # TEMA calculation
    def tema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3

    # MA function for TDFI
    def ma_tdfi(mode, src, length):
        if mode == 'ema':
            return src.ewm(span=length, adjust=False).mean()
        elif mode == 'wma':
            return src.rolling(window=length).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True)
        elif mode == 'swma':
            return src.rolling(window=4).mean()
        elif mode == 'vwma':
            return pd.Series([np.average(src.iloc[i-length:i], weights=df['volume'].iloc[i-length:i]) if i >= length else np.nan for i in range(len(src))])
        elif mode == 'hull':
            half_len = int(length / 2)
            wma_half = src.rolling(window=half_len).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True)
            wma_full = src.rolling(window=length).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True)
            return (2 * wma_half - wma_full).rolling(window=int(np.sqrt(length))).mean()
        elif mode == 'tema':
            return tema(src, length)
        else:
            return src.rolling(length).mean()

    price_tdfi = df['close'] * 1000
    mma_tdfi = ma_tdfi(mmaModeTDFI, price_tdfi, mmaLengthTDFI)
    smma_tdfi = ma_tdfi(smmaModeTDFI, mma_tdfi, smmaLengthTDFI)

    impetmma_tdfi = mma_tdfi - mma_tdfi.shift(1)
    impetsmma_tdfi = smma_tdfi - smma_tdfi.shift(1)
    divma_tdfi = np.abs(mma_tdfi - smma_tdfi)
    averimpet_tdfi = (impetmma_tdfi + impetsmma_tdfi) / 2
    tdf_tdfi = np.power(divma_tdfi, 1) * np.power(averimpet_tdfi, nLengthTDFI)

    # Rolling highest for normalization
    rolling_high = tdf_tdfi.abs().rolling(window=lookbackTDFI * nLengthTDFI).max()
    signal_tdfi = tdf_tdfi / rolling_high

    # TDFI signals
    signal_long_tdfi = signal_tdfi > filterHighTDFI
    signal_short_tdfi = signal_tdfi < filterLowTDFI

    final_long_signal_tdfi = signal_long_tdfi
    final_short_signal_tdfi = signal_short_tdfi

    # === HEVs (HawkEye Volume) Parameters ===
    length_hev = 200
    range_1 = df['high'] - df['low']
    range_avg = range_1.rolling(length_hev).mean()
    hv_ma = 20
    durchschnitt = df['volume'].rolling(hv_ma).mean()
    volume_a = df['volume'].rolling(length_hev).mean()
    divisor = 3.6

    high1 = df['high'].shift(1)
    low1 = df['low'].shift(1)
    mid1 = ((df['high'] + df['low']) / 2).shift(1)

    u1 = mid1 + (high1 - low1) / divisor
    d1 = mid1 - (high1 - low1) / divisor

    r_enabled1 = (range_1 > range_avg) & (df['close'] < d1) & (df['volume'] > volume_a)
    r_enabled2 = df['close'] < mid1
    r_enabled = r_enabled1 | r_enabled2

    g_enabled1 = df['close'] > mid1
    g_enabled2 = (range_1 > range_avg) & (df['close'] > u1) & (df['volume'] > volume_a)
    g_enabled3 = (df['high'] > high1) & (range_1 < range_avg / 1.5) & (df['volume'] < volume_a)
    g_enabled4 = (df['low'] < low1) & (range_1 < range_avg / 1.5) & (df['volume'] > volume_a)
    g_enabled = g_enabled1 | g_enabled2 | g_enabled3 | g_enabled4

    basic_long_hev_condition = g_enabled & (df['volume'] > durchschnitt)
    basic_short_hev_condition = r_enabled & (df['volume'] > durchschnitt)

    hev_signals_long = basic_long_hev_condition
    hev_signals_short = basic_short_hev_condition

    hev_signals_long_final = hev_signals_long
    hev_signals_short_final = hev_signals_short

    # === Combined Entry Conditions ===
    long_condition = (finalLongSignalT3) & (signal_tdfi > filterHighTDFI) & (basic_long_hev_condition)
    short_condition = (finalShortSignalT3) & (signal_tdfi < filterLowTDFI) & (basic_short_hev_condition)

    # === Generate Entries ===
    for i in range(len(df)):
        # Skip bars with NaN in key indicators
        if pd.isna(t3_25.iloc[i]) or pd.isna(t3_100.iloc[i]) or pd.isna(t3_200.iloc[i]):
            continue
        if pd.isna(signal_tdfi.iloc[i]):
            continue
        if pd.isna(basic_long_hev_condition.iloc[i]) or pd.isna(basic_short_hev_condition.iloc[i]):
            continue

        entry_price = df['close'].iloc[i]

        # Long entry
        if long_condition.iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

        # Short entry
        if short_condition.iloc[i]:
            results.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

    return results