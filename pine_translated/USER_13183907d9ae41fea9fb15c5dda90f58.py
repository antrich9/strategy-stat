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
    
    # === E2PSS Parameters ===
    useE2PSS = True
    inverseE2PSS = False
    PriceE2PSS = (df['high'] + df['low']) / 2  # hl2
    PeriodE2PSS = 15
    
    pi = 2 * np.arcsin(1)
    
    a1 = np.exp(-1.414 * pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    Filt2 = np.zeros(len(df))
    TriggerE2PSS = np.zeros(len(df))
    
    Filt2[0] = PriceE2PSS.iloc[0]
    Filt2[1] = PriceE2PSS.iloc[1]
    
    for i in range(2, len(df)):
        Filt2[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
    
    TriggerE2PSS = pd.Series(Filt2).shift(1)
    
    signalLongE2PSS = ~useE2PSS | (Filt2 > TriggerE2PSS)
    signalShortE2PSS = ~useE2PSS | (Filt2 < TriggerE2PSS)
    
    signalLongE2PSSFinal = ~inverseE2PSS & signalLongE2PSS | inverseE2PSS & signalShortE2PSS
    signalShortE2PSSFinal = ~inverseE2PSS & signalShortE2PSS | inverseE2PSS & signalLongE2PSS
    
    # === TDFI Parameters ===
    useTDFI = True
    crossTDFI = True
    inverseTDFI = True
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
    
    def wma(src, length):
        weights = np.arange(1, length + 1)
        return src.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    def swma(src):
        return src.rolling(8).mean()
    
    def vwma(src, length):
        return (src * df['volume']).rolling(length).sum() / df['volume'].rolling(length).sum()
    
    def hull(src, length):
        half = int(length / 2)
        sqrt_len = int(np.round(np.sqrt(length)))
        wma_half = wma(src, half)
        wma_full = wma(src, length)
        diff = 2 * wma_half - wma_full
        return wma(diff, sqrt_len)
    
    def ma_tdfi(mode, src, length):
        if mode == 'ema':
            return src.ewm(span=length, adjust=False).mean()
        elif mode == 'wma':
            return wma(src, length)
        elif mode == 'swma':
            return swma(src)
        elif mode == 'vwma':
            return vwma(src, length)
        elif mode == 'hull':
            return hull(src, length)
        elif mode == 'tema':
            return tema_tdfi(src, length)
        else:
            return src.rolling(length).mean()
    
    mma_tdfi = ma_tdfi(mmaModeTDFI, priceTDFI * 1000, mmaLengthTDFI)
    smma_tdfi = ma_tdfi(smmaModeTDFI, mma_tdfi, smmaLengthTDFI)
    
    impetmma_tdfi = mma_tdfi - mma_tdfi.shift(1)
    impetsmma_tdfi = smma_tdfi - smma_tdfi.shift(1)
    divma_tdfi = (mma_tdfi - smma_tdfi).abs()
    averimpet_tdfi = (impetmma_tdfi + impetsmma_tdfi) / 2
    tdf_tdfi = (divma_tdfi ** 1) * (averimpet_tdfi ** nLengthTDFI)
    
    highest_abs_tdf = tdf_tdfi.abs().rolling(lookbackTDFI * nLengthTDFI).max()
    signal_tdfi = tdf_tdfi / highest_abs_tdf
    
    if crossTDFI:
        signal_long_tdfi = (signal_tdfi > signal_tdfi.shift(1)) & (signal_tdfi > filterHighTDFI)
        signal_short_tdfi = (signal_tdfi < signal_tdfi.shift(1)) & (signal_tdfi < filterLowTDFI)
    else:
        signal_long_tdfi = signal_tdfi > filterHighTDFI
        signal_short_tdfi = signal_tdfi < filterLowTDFI
    
    signal_long_tdfi = useTDFI & signal_long_tdfi if crossTDFI else useTDFI & signal_long_tdfi
    signal_short_tdfi = useTDFI & signal_short_tdfi if crossTDFI else useTDFI & signal_short_tdfi
    
    final_long_signal_tdfi = ~inverseTDFI & signal_long_tdfi if not inverseTDFI else signal_short_tdfi
    final_short_signal_tdfi = inverseTDFI & signal_long_tdfi if inverseTDFI else signal_short_tdfi
    
    # === Stiffness Parameters ===
    useStiffness = False
    maLengthStiffness = 100
    stiffLength = 60
    stiffSmooth = 3
    thresholdStiffness = 90
    
    sma_stiff = close.rolling(maLengthStiffness).mean()
    std_stiff = close.rolling(maLengthStiffness).std()
    boundStiffness = sma_stiff - 0.2 * std_stiff
    
    sumAbove = (close > boundStiffness).astype(int).rolling(stiffLength).sum()
    stiffness = (sumAbove * 100 / stiffLength).ewm(span=stiffSmooth, adjust=False).mean()
    
    signal_stiffness = True if not useStiffness else stiffness > thresholdStiffness
    
    # === Entry Conditions ===
    long_condition = signalLongE2PSSFinal & (final_long_signal_tdfi == 1) & (signal_stiffness == 1)
    short_condition = signalShortE2PSSFinal & (final_short_signal_tdfi == -1) & (signal_stiffness == -1)
    
    # === Generate Entries ===
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_condition.iloc[i]:
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            entry_price = close.iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries