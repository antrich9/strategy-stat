import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    open_price = df['open']
    high = df['high']
    low = df['low']
    
    # JMA parameters
    lengthjmaJMA = 7
    phasejmaJMA = 50
    powerjmaJMA = 2
    srcjmaJMA = close
    
    phasejmaJMARatiojmaJMA = 0.5 if phasejmaJMA < -100 else (2.5 if phasejmaJMA > 100 else phasejmaJMA / 100 + 1.5)
    betajmaJMA = 0.45 * (lengthjmaJMA - 1) / (0.45 * (lengthjmaJMA - 1) + 2)
    alphajmaJMA = betajmaJMA ** powerjmaJMA
    
    e0JMA = pd.Series(0.0, index=df.index)
    e1JMA = pd.Series(0.0, index=df.index)
    e2JMA = pd.Series(0.0, index=df.index)
    jmaJMA = pd.Series(0.0, index=df.index)
    
    for i in range(len(df)):
        if i == 0:
            e0JMA.iloc[i] = (1 - alphajmaJMA) * srcjmaJMA.iloc[i]
            e1JMA.iloc[i] = 0.0
            e2JMA.iloc[i] = 0.0
            jmaJMA.iloc[i] = 0.0
        else:
            e0JMA.iloc[i] = (1 - alphajmaJMA) * srcjmaJMA.iloc[i] + alphajmaJMA * (e0JMA.iloc[i-1] if not np.isnan(e0JMA.iloc[i-1]) else srcjmaJMA.iloc[i-1])
            e1JMA.iloc[i] = (srcjmaJMA.iloc[i] - e0JMA.iloc[i]) * (1 - betajmaJMA) + betajmaJMA * (e1JMA.iloc[i-1] if not np.isnan(e1JMA.iloc[i-1]) else 0.0)
            e2JMA.iloc[i] = ((e0JMA.iloc[i] + phasejmaJMARatiojmaJMA * e1JMA.iloc[i] - (jmaJMA.iloc[i-1] if not np.isnan(jmaJMA.iloc[i-1]) else 0.0)) * (1 - alphajmaJMA) ** 2 + alphajmaJMA ** 2 * (e2JMA.iloc[i-1] if not np.isnan(e2JMA.iloc[i-1]) else 0.0))
            jmaJMA.iloc[i] = e2JMA.iloc[i] + (jmaJMA.iloc[i-1] if not np.isnan(jmaJMA.iloc[i-1]) else 0.0)
    
    jmaJMA = jmaJMA.replace(0, np.nan)
    
    signalmaJMALong = (jmaJMA > jmaJMA.shift(1)) & (close > jmaJMA)
    signalmaJMAShort = (jmaJMA < jmaJMA.shift(1)) & (close < jmaJMA)
    
    finalLongSignalJMA = signalmaJMAShort
    finalShortSignalJMA = signalmaJMALong
    
    # E2PSS
    PeriodE2PSS = 15
    PriceE2PSS = (high + low) / 2
    
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * 3.14159 / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    Filt2 = pd.Series(0.0, index=df.index)
    TriggerE2PSS = pd.Series(0.0, index=df.index)
    
    for i in range(len(df)):
        if i < 3:
            Filt2.iloc[i] = PriceE2PSS.iloc[i]
        else:
            prev_filt2_1 = Filt2.iloc[i-1] if not np.isnan(Filt2.iloc[i-1]) else 0.0
            prev_filt2_2 = Filt2.iloc[i-2] if not np.isnan(Filt2.iloc[i-2]) else 0.0
            Filt2.iloc[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * prev_filt2_1 + coef3 * prev_filt2_2
        TriggerE2PSS.iloc[i] = Filt2.iloc[i-1] if i > 0 else 0.0
    
    Filt2 = Filt2.replace(0, np.nan)
    TriggerE2PSS = TriggerE2PSS.replace(0, np.nan)
    
    signalLongE2PSS = Filt2 > TriggerE2PSS
    signalShortE2PSS = Filt2 < TriggerE2PSS
    
    signalLongE2PSSFinal = signalLongE2PSS
    signalShortE2PSSFinal = signalShortE2PSS
    
    # Triple T3
    factorT3 = 0.7
    
    def gdT3(src, length):
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
    
    t3_25 = gdT3(close, 25)
    t3_100 = gdT3(close, 100)
    t3_200 = gdT3(close, 200)
    
    longConditionIndiT3 = (close > t3_25) & (close > t3_100) & (close > t3_200)
    shortConditionIndiT3 = (close < t3_25) & (close < t3_100) & (close < t3_200)
    longConditionIndiT3MA = (t3_100 < t3_25) & (t3_200 < t3_100)
    shortConditionIndiT3MA = (t3_100 > t3_25) & (t3_200 > t3_100)
    
    signalEntryLongT3 = longConditionIndiT3 & longConditionIndiT3MA
    signalEntryShortT3 = shortConditionIndiT3 & shortConditionIndiT3MA
    
    signalEntryLongT3 = signalEntryLongT3 & (~signalEntryLongT3.shift(1).fillna(False))
    signalEntryShortT3 = signalEntryShortT3 & (~signalEntryShortT3.shift(1).fillna(False))
    
    finalLongSignalT3 = signalEntryShortT3
    finalShortSignalT3 = signalEntryLongT3
    
    # Trendilo
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    pct_change = close.diff(trendilo_smooth) / close.shift(trendilo_smooth) * 100
    
    window = trendilo_length
    m = 2 / (trendilo_offset * (window - 1) + 1)
    alp = -trendilo_sigma ** 2 * 0.5
    w = np.exp(np.arange(window) ** 2 * alp)
    w = w / w.sum()
    
    avg_pct_change = pd.Series(index=df.index, dtype=float)
    for i in range(window - 1, len(df)):
        vals = pct_change.iloc[i-window+1:i+1].values
        avg_pct_change.iloc[i] = np.sum(vals * w)
    avg_pct_change = avg_pct_change.fillna(0)
    
    rms_arr = np.sqrt(avg_pct_change.iloc[trendilo_length-1:].rolling(trendilo_length).apply(lambda x: (x ** 2).sum() / trendilo_length).fillna(0).values)
    rms = pd.Series(index=df.index, dtype=float)
    for i in range(len(rms_arr)):
        rms.iloc[i + trendilo_length - 1] = trendilo_bmult * rms_arr[i]
    rms = rms.fillna(0)
    
    trendilo_dir = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if avg_pct_change.iloc[i] > rms.iloc[i]:
            trendilo_dir.iloc[i] = 1
        elif avg_pct_change.iloc[i] < -rms.iloc[i]:
            trendilo_dir.iloc[i] = -1
        else:
            trendilo_dir.iloc[i] = 0
    trendilo_dir = trendilo_dir.fillna(0).astype(int)
    
    finalLongSignalTrendilo = trendilo_dir == 1
    finalShortSignalTrendilo = trendilo_dir == -1
    
    # TDFI
    lookbackTDFI = 13
    mmaLengthTDFI = 13
    mmaModeTDFI = 'ema'
    smmaLengthTDFI = 13
    smmaModeTDFI = 'ema'
    nLengthTDFI = 3
    filterHighTDFI = 0.05
    filterLowTDFI = -0.05
    priceTDFI = close
    
    def tema_tdfi(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        ema3 = ema2.ewm(span=length, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3
    
    def ma_tdfi(mode, src, length):
        if mode == 'ema':
            return src.ewm(span=length, adjust=False).mean()
        elif mode == 'wma':
            weights = np.arange(1, length + 1)
            return src.rolling(length).apply(lambda x: np.sum(weights * x) / weights.sum(), raw=True)
        elif mode == 'swma':
            return src.rolling(4).apply(lambda x: x.iloc[0]*0.25 + x.iloc[1]*0.25 + x.iloc[2]*0.25 + x.iloc[3]*0.25 if len(x)==4 else np.nan, raw=True)
        elif mode == 'vwma':
            return (src * df['volume']).rolling(length).sum() / df['volume'].rolling(length).sum()
        elif mode == 'hull':
            wma_half = src.rolling(int(length/2)).apply(lambda x: np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1)), raw=True)
            hull = 2 * wma_half - src.rolling(length).apply(lambda x: np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1)), raw=True)
            return hull.rolling(int(np.sqrt(length))).apply(lambda x: np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1)), raw=True)
        elif mode == 'tema':
            return tema_tdfi(src, length)
        else:
            return src.rolling(length).mean()
    
    mma_tdfi = ma_tdfi(mmaModeTDFI, priceTDFI * 1000, mmaLengthTDFI)
    smma_tdfi = ma_tdfi(smmaModeTDFI, mma_tdfi, smmaLengthTDFI)
    impetmma = mma_tdfi - mma_tdfi.shift(1)
    impetsmma = smma_tdfi - smma_tdfi.shift(1)
    divma = np.abs(mma_tdfi - smma_tdfi)
    averimpet = (impetmma + impetsmma) / 2
    tdf_raw = (divma ** 1) * (averimpet ** nLengthTDFI)
    
    tdfi_val = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        lookback = lookbackTDFI * nLengthTDFI
        if i >= lookback:
            window_vals = tdf_raw.iloc[i-lookback+1:i+1].fillna(0).values
            max_val = np.max(np.abs(window_vals))
            if max_val != 0 and not np.isnan(max_val):
                tdfi_val.iloc[i] = tdf_raw.iloc[i] / max_val
            else:
                tdfi_val.iloc[i] = 0
        else:
            tdfi_val.iloc[i] = 0
    tdfi_val = tdfi_val.replace(0, np.nan)
    
    signalLongTDFI_raw = tdfi_val > filterHighTDFI
    signalShortTDFI_raw = tdfi_val < filterLowTDFI
    signalLongTDFI_crossover = (tdfi_val > filterHighTDFI) & (tdfi_val.shift(1) <= filterHighTDFI)
    signalShortTDFI_crossunder = (tdfi_val < filterLowTDFI) & (tdfi_val.shift(1) >= filterLowTDFI)
    
    signalLongTDFI = signalLongTDFI_crossover
    signalShortTDFI = signalShortTDFI_crossunder
    
    finalLongSignalTDFI = signalShortTDFI
    finalShortSignalTDFI = signalLongTDFI
    
    # Stiffness
    maLengthStiffness = 100
    stiffLength = 60
    stiffSmooth = 3
    thresholdStiffness = 90
    
    boundStiffness = close.rolling(maLengthStiffness).mean() - 0.2 * close.rolling(maLengthStiffness).std()
    sumAbove = (close > boundStiffness).rolling(stiffLength).sum()
    stiffness = (sumAbove * 100 / stiffLength).ewm(span=stiffSmooth, adjust=False).mean()
    
    signalStiffness = stiffness > thresholdStiffness
    
    # TTMS
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    BB_basis = close.rolling(length_TTMS).mean()
    dev_TTMS = BB_mult_TTMS * close.rolling(length_TTMS).std()
    BB_upper = BB_basis
    BB_lower = BB_basis
    
    keltner_mult = 1.5
    keltner_basis = close.ewm(span=length_TTMS, adjust=False).mean()
    keltner_upper = keltner_basis + keltner_mult * close.ewm(span=length_TTMS, adjust=False).mean()
    
    long_TTMS = BB_upper < keltner_upper
    short_TTMS = BB_upper > keltner_upper
    
    # Combined entries
    long_condition = (finalLongSignalJMA & signalLongE2PSSFinal & finalLongSignalT3 & finalLongSignalTrendilo & finalShortSignalTDFI & signalStiffness & long_TTMS)
    short_condition = (finalShortSignalJMA & signalShortE2PSSFinal & finalShortSignalT3 & finalShortSignalTrendilo & finalLongSignalTDFI & signalStiffness & short_TTMS)
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i < 3:
            continue
        if np.isnan(jmaJMA.iloc[i]) or np.isnan(t3_25.iloc[i]) or np.isnan(t3_100.iloc[i]) or np.isnan(t3_200.iloc[i]):
            continue
        if np.isnan(tdfi_val.iloc[i]) or np.isnan(stiffness.iloc[i]):
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
        elif short_condition.iloc[i]:
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