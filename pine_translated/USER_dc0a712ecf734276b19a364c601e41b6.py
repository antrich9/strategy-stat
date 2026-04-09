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
    allowLong = True
    allowShort = True
    requireAllSignals = True
    lookbackTDFI = 13
    mmaLengthTDFI = 13
    mmaModeTDFI = 'ema'
    smmaLengthTDFI = 13
    smmaModeTDFI = 'ema'
    nLengthTDFI = 3
    filterHighTDFI = 0.05
    filterLowTDFI = -0.05
    lengthDMH = 10
    useTDFI = True
    crossTDFI = True
    inverseTDFI = False
    useDMH = True
    dmhLongCondition = "Rising"
    dmhShortCondition = "Falling"
    useADX = True
    adxThreshold = 25
    adx_len = 14
    di_len = 14
    useDateFilter = False
    startDate = None
    endDate = None

    close = df['close']
    high = df['high']
    low = df['low']

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
            def wma_helper(series):
                return series.rolling(length).apply(lambda x: (weights[:len(x)] * x).sum() / weights[:len(x)].sum(), raw=True)
            return pd.Series(wma_helper(src), index=src.index)
        elif mode == 'swma':
            return src.rolling(4).mean()
        elif mode == 'vwma':
            return pd.Series(np.nan, index=src.index)
        elif mode == 'hull':
            wma_half = src.ewm(span=int(length/2), adjust=False).mean()
            wma_full = src.ewm(span=length, adjust=False).mean()
            hull = 2 * wma_half - wma_full
            sqrt_len = int(np.sqrt(length))
            return hull.ewm(span=sqrt_len, adjust=False).mean()
        elif mode == 'tema':
            return tema_tdfi(src, length)
        else:
            return src.rolling(length).mean()

    def tdfi_tdfi():
        price = close * 1000
        mma_tdfi = ma_tdfi(mmaModeTDFI, price, mmaLengthTDFI)
        smma_tdfi = ma_tdfi(smmaModeTDFI, mma_tdfi, smmaLengthTDFI)
        impetmma = mma_tdfi - mma_tdfi.shift(1)
        impetsmma = smma_tdfi - smma_tdfi.shift(1)
        divma = (mma_tdfi - smma_tdfi).abs()
        averimpet = (impetmma + impetsmma) / 2
        lookback = lookbackTDFI * nLengthTDFI
        abs_tdf = (divma * averimpet.pow(nLengthTDFI)).abs()
        highest_tdf = abs_tdf.rolling(lookback).max()
        tdf = abs_tdf / highest_tdf
        return tdf

    signal_tdfi = tdfi_tdfi()

    signal_long_tdfi = pd.Series(True, index=df.index)
    signal_short_tdfi = pd.Series(True, index=df.index)
    if useTDFI:
        if crossTDFI:
            signal_long_tdfi = (signal_tdfi > signal_tdfi.shift(1)) & (signal_tdfi.shift(1) <= filterHighTDFI) & (signal_tdfi > filterHighTDFI)
            signal_short_tdfi = (signal_tdfi < signal_tdfi.shift(1)) & (signal_tdfi.shift(1) >= filterLowTDFI) & (signal_tdfi < filterLowTDFI)
        else:
            signal_long_tdfi = signal_tdfi > filterHighTDFI
            signal_short_tdfi = signal_tdfi < filterLowTDFI

    final_long_tdfi = signal_short_tdfi if inverseTDFI else signal_long_tdfi
    final_short_tdfi = signal_long_tdfi if inverseTDFI else signal_short_tdfi

    def rma(src, length):
        alpha = 1.0 / length
        result = pd.Series(np.nan, index=src.index)
        result.iloc[0] = src.iloc[0]
        for i in range(1, len(src)):
            result.iloc[i] = src.iloc[i] * alpha + result.iloc[i-1] * (1 - alpha)
        return result

    def dmh_calc(period):
        up_move = high - high.shift(1)
        dn_move = -(low - low.shift(1))
        p_dm = ((up_move > dn_move) & (up_move > 0)).astype(float) * up_move
        m_dm = ((dn_move > up_move) & (dn_move > 0)).astype(float) * dn_move
        dm_diff = p_dm - m_dm
        hann_smoothed = rma(dm_diff, period)
        return hann_smoothed

    signal_dmh = dmh_calc(lengthDMH)
    dmh_rising = signal_dmh > signal_dmh.shift(1)
    dmh_falling = signal_dmh < signal_dmh.shift(1)
    dmh_above_zero = signal_dmh > 0
    dmh_below_zero = signal_dmh < 0

    dmh_long_signal = pd.Series(True, index=df.index)
    dmh_short_signal = pd.Series(True, index=df.index)
    if useDMH:
        if dmhLongCondition == "Rising":
            dmh_long_signal = dmh_rising
        elif dmhLongCondition == "Above Zero":
            dmh_long_signal = dmh_above_zero
        else:
            dmh_long_signal = dmh_rising & dmh_above_zero

        if dmhShortCondition == "Falling":
            dmh_short_signal = dmh_falling
        elif dmhShortCondition == "Below Zero":
            dmh_short_signal = dmh_below_zero
        else:
            dmh_short_signal = dmh_falling & dmh_below_zero

    def wilder_smooth(src, length):
        alpha = 1.0 / length
        result = pd.Series(np.nan, index=src.index)
        result.iloc[0] = src.iloc[0]
        for i in range(1, len(src)):
            result.iloc[i] = src.iloc[i] * alpha + result.iloc[i-1] * (1 - alpha)
        return result

    def calc_adx(dilen, adxlen):
        up = high.diff()
        down = -low.diff()
        plus_dm = ((up > down) & (up > 0)).astype(float) * up
        minus_dm = ((down > up) & (down > 0)).astype(float) * down
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        truerange = wilder_smooth(tr, dilen)
        plus_dm_smooth = wilder_smooth(plus_dm, dilen)
        minus_dm_smooth = wilder_smooth(minus_dm, dilen)
        plus = (plus_dm_smooth / truerange * 100).fillna(0)
        minus = (minus_dm_smooth / truerange * 100).fillna(0)
        sum_dm = plus + minus
        adx = (pd.Series(np.abs(plus - minus) / sum_dm.replace(0, 1), index=df.index).ewm(span=adxlen, adjust=False).mean() * 100)
        return adx

    adx_value = calc_adx(di_len, adx_len)
    adx_filter = adx_value >= adxThreshold if useADX else pd.Series(True, index=df.index)

    long_signal = pd.Series(False, index=df.index)
    short_signal = pd.Series(False, index=df.index)
    if requireAllSignals:
        long_signal = final_long_tdfi & dmh_long_signal & adx_filter
        short_signal = final_short_tdfi & dmh_short_signal & adx_filter
    else:
        long_signal = (final_long_tdfi | dmh_long_signal) & adx_filter
        short_signal = (final_short_tdfi | dmh_short_signal) & adx_filter

    in_date_range = pd.Series(True, index=df.index)
    if useDateFilter:
        timestamps = df['time']
        in_date_range = (timestamps >= startDate) & (timestamps <= endDate)

    enter_long = allowLong & long_signal & in_date_range
    enter_short = allowShort & short_signal & in_date_range

    entries = []
    trade_num = 1
    position_open = False

    for i in range(1, len(df)):
        if position_open:
            if enter_long.iloc[i]:
                pass
            elif enter_short.iloc[i]:
                pass
        else:
            if enter_long.iloc[i]:
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
                position_open = True
            elif enter_short.iloc[i]:
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
                position_open = True

    return entries