import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']
    
    factor = 0.7
    useT3 = True
    signalTypeT3 = 'MA + Price'
    crossOnlyT3 = True
    inverseT3 = False
    
    useTDFI = True
    crossTDFI = True
    inverseTDFI = True
    lookbackTDFI = 13
    mmaLengthTDFI = 13
    smmaLengthTDFI = 13
    nLengthTDFI = 3
    filterHighTDFI = 0.05
    filterLowTDFI = -0.05
    
    length_TTMS = 20
    redGreen_TTMS = True
    cross_TTMS = True
    inverse_TTMS = False
    highlightMovements_TTMS = True
    
    def gd(src, length):
        e1 = src.ewm(span=length, adjust=False).mean()
        e2 = e1.ewm(span=length, adjust=False).mean()
        e3 = e2.ewm(span=length, adjust=False).mean()
        e4 = e3.ewm(span=length, adjust=False).mean()
        e5 = e4.ewm(span=length, adjust=False).mean()
        e6 = e5.ewm(span=length, adjust=False).mean()
        c1 = -factor ** 3
        c2 = 3 * factor ** 2 + 3 * factor ** 3
        c3 = -6 * factor ** 2 - 3 * factor - 3 * factor ** 3
        c4 = 1 + 3 * factor + factor ** 2 + 3 * factor ** 2
        return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    
    t3_25 = gd(close, 25)
    t3_100 = gd(close, 100)
    t3_200 = gd(close, 200)
    
    longCondIndiT3 = close > t3_25 & close > t3_100 & close > t3_200
    shortCondIndiT3 = close < t3_25 & close < t3_100 & close < t3_200
    longCondIndiT3MA = t3_100 < t3_25 & t3_200 < t3_100
    shortCondIndiT3MA = t3_100 > t3_25 & t3_200 > t3_100
    
    if signalTypeT3 == 'MA + Price':
        signalLongT3 = longCondIndiT3 & longCondIndiT3MA
        signalShortT3 = shortCondIndiT3 & shortCondIndiT3MA
    elif signalTypeT3 == 'MA Only':
        signalLongT3 = longCondIndiT3MA
        signalShortT3 = shortCondIndiT3MA
    else:
        signalLongT3 = longCondIndiT3
        signalShortT3 = shortCondIndiT3
    
    signalLongT3_raw = signalLongT3.copy()
    signalShortT3_raw = signalShortT3.copy()
    
    if crossOnlyT3:
        signalLongT3 = signalLongT3 & ~signalLongT3.shift(1, fill_value=False)
        signalShortT3 = signalShortT3 & ~signalShortT3.shift(1, fill_value=False)
    
    finalLongT3 = signalShortT3 if inverseT3 else signalLongT3
    finalShortT3 = signalLongT3 if inverseT3 else signalShortT3
    
    price_tdfi = close * 1000
    mma_tdfi = price_tdfi.ewm(span=mmaLengthTDFI, adjust=False).mean()
    smma_tdfi = mma_tdfi.ewm(span=smmaLengthTDFI, adjust=False).mean()
    impet_mma = mma_tdfi.diff()
    impet_smma = smma_tdfi.diff()
    div_ma = (mma_tdfi - smma_tdfi).abs()
    aver_impet = (impet_mma + impet_smma) / 2
    tdf_raw = div_ma * (aver_impet ** nLengthTDFI)
    highest_val = tdf_raw.abs().rolling(window=lookbackTDFI * nLengthTDFI).max()
    signal_tdfi = tdf_raw / highest_val
    
    signalLongTDFI_base = signal_tdfi > filterHighTDFI
    signalShortTDFI_base = signal_tdfi < filterLowTDFI
    
    if crossTDFI:
        signalLongTDFI = (signal_tdfi > filterHighTDFI) & (signal_tdfi.shift(1, fill_value=0) <= filterHighTDFI)
        signalShortTDFI = (signal_tdfi < filterLowTDFI) & (signal_tdfi.shift(1, fill_value=0) >= filterLowTDFI)
    else:
        signalLongTDFI = signalLongTDFI_base
        signalShortTDFI = signalShortTDFI_base
    
    finalLongTDFI = signalShortTDFI if inverseTDFI else signalLongTDFI
    finalShortTDFI = signalLongTDFI if inverseTDFI else signalShortTDFI
    
    tr = high - low
    tr = pd.concat([tr, high.shift(1) - low, high - low.shift(1)], axis=1).max(axis=1)
    
    BB_basis = close.rolling(length_TTMS).mean()
    dev_bb = close.rolling(length_TTMS).std()
    BB_upper = BB_basis + 2.0 * dev_bb
    BB_lower = BB_basis - 2.0 * dev_bb
    
    dev_kc = tr.rolling(length_TTMS).mean()
    KC_upper_high = BB_basis + dev_kc * 1.0
    KC_lower_high = BB_basis - dev_kc * 1.0
    KC_upper_mid = BB_basis + dev_kc * 1.5
    KC_lower_mid = BB_basis - dev_kc * 1.5
    KC_upper_low = BB_basis + dev_kc * 2.0
    KC_lower_low = BB_basis - dev_kc * 2.0
    
    NoSqz = (BB_lower < KC_lower_low) | (BB_upper > KC_upper_low)
    
    avg_hl = (high.rolling(length_TTMS).max() + low.rolling(length_TTMS).min()) / 2
    avg_all = (avg_hl + close.rolling(length_TTMS).mean()) / 2
    dev_val = close - avg_all
    
    mom = pd.Series(index=close.index, dtype=float)
    x = np.arange(length_TTMS)
    for i in range(length_TTMS - 1, len(close)):
        vals = dev_val.iloc[i - length_TTMS + 1:i + 1].values
        if len(vals) == length_TTMS and not np.any(np.isnan(vals)):
            slope = np.polyfit(x, vals, 1)[0]
            mom.iloc[i] = slope
        else:
            mom.iloc[i] = np.nan
    
    iff_1 = 1
    iff_2 = -1
    TTMS_signals = pd.Series(np.where(mom > 0, iff_1, iff_2), index=close.index)
    
    if redGreen_TTMS:
        basicLongCond = TTMS_signals == 1
        basicShortCond = TTMS_signals == -1
    else:
        basicLongCond = TTMS_signals > 0
        basicShortCond = TTMS_signals < 0
    
    basicLongCond = basicLongCond.fillna(False).astype(bool)
    basicShortCond = basicShortCond.fillna(False).astype(bool)
    
    long_squeeze = (highlightMovements_TTMS & NoSqz) & basicLongCond
    short_squeeze = (highlightMovements_TTMS & NoSqz) & basicShortCond
    
    if cross_TTMS:
        long_cross = ~long_squeeze.shift(1, fill_value=False) & long_squeeze
        short_cross = ~short_squeeze.shift(1, fill_value=False) & short_squeeze
    else:
        long_cross = long_squeeze.copy()
        short_cross = short_squeeze.copy()
    
    finalLongTTMS = short_cross if inverse_TTMS else long_cross
    finalShortTTMS = long_cross if inverse_TTMS else short_cross
    
    finalLongT3 = finalLongT3.fillna(False).astype(bool)
    finalShortT3 = finalShortT3.fillna(False).astype(bool)
    finalLongTDFI = finalLongTDFI.fillna(False).astype(bool)
    finalShortTDFI = finalShortTDFI.fillna(False).astype(bool)
    finalLongTTMS = finalLongTTMS.fillna(False).astype(bool)
    finalShortTTMS = finalShortTTMS.fillna(False).astype(bool)
    basicLongCond = basicLongCond.fillna(False).astype(bool)
    basicShortCond = basicShortCond.fillna(False).astype(bool)
    
    long_entry = finalLongT3 & finalLongTDFI & basicLongCond
    short_entry = finalShortT3 & finalShortTDFI & basicShortCond
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
    
    return entries