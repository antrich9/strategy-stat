import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values
    timestamps = df['time'].values
    n = len(df)
    
    atrLength = 14
    rfRngPer = 200
    rfSmoothPer = 200
    rfRngQty = 2.618
    rfFilterType = 'Type 1'
    rfSmoothRange = True
    rfAvVals = False
    rfAvSamples = 2
    
    lookbackTDFI = 13
    mmaLengthTDFI = 13
    smmaLengthTDFI = 13
    nLengthTDFI = 3
    filterHighTDFI = 0.0
    filterLowTDFI = 0.0
    mmaModeTDFI = 'ema'
    smmaModeTDFI = 'ema'
    useTDFI = True
    crossTDFI = False
    inverseTDFI = False
    
    lengthDMH = 10
    dmhLongCondition = 'Above Zero'
    dmhShortCondition = 'Below Zero'
    useDMH = True
    
    adx_len = 14
    di_len = 14
    adxThreshold = 25
    useADX = True
    useADXMA = True
    adxMALength = 75
    
    allowLong = True
    allowShort = True
    requireAllSignals = True
    
    def calc_tr(h, l, cl):
        tr1 = h - l
        tr2 = np.abs(h - np.roll(cl, 1))
        tr3 = np.abs(l - np.roll(cl, 1))
        tr2[0] = 0
        tr3[0] = 0
        return np.maximum.reduce([tr1, tr2, tr3])
    
    def calc_atr(h, l, cl, length):
        tr = calc_tr(h, l, cl)
        atr = np.zeros(n)
        atr[length-1] = np.mean(tr[:length])
        for i in range(length, n):
            atr[i] = (atr[i-1] * (length - 1) + tr[i]) / length
        return atr
    
    def calc_tema(src, length):
        ema1 = pd.Series(src).ewm(span=length, adjust=False).mean().values
        ema2 = pd.Series(ema1).ewm(span=length, adjust=False).mean().values
        ema3 = pd.Series(ema2).ewm(span=length, adjust=False).mean().values
        return 3 * ema1 - 3 * ema2 + ema3
    
    def ma_func(mode, src, length):
        if mode == 'ema':
            return pd.Series(src).ewm(span=length, adjust=False).mean().values
        elif mode == 'wma':
            weights = np.arange(1, length + 1)
            return pd.Series(src).rolling(length).apply(lambda x: np.sum(weights * x) / np.sum(weights), raw=True).values
        elif mode == 'swma':
            return pd.Series(src).rolling(8).apply(lambda x: np.sum(x * np.array([-1, 2, 3, 4, 4, 3, 2, -1])) / 10, raw=True).values
        elif mode == 'vwma':
            return pd.Series(src * volume).rolling(length).sum() / pd.Series(volume).rolling(length).sum()
        elif mode == 'hull':
            half = int(length / 2)
            wma1 = pd.Series(src).rolling(half).apply(lambda x: np.sum(np.arange(1, half+1) * x) / np.sum(np.arange(1, half+1)), raw=True).values
            wma2 = pd.Series(src).rolling(length).apply(lambda x: np.sum(np.arange(1, length+1) * x) / np.sum(np.arange(1, length+1)), raw=True).values
            hull = pd.Series(2 * wma1 - wma2).rolling(int(np.sqrt(length))).apply(lambda x: np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1)), raw=True).values
            return hull
        elif mode == 'tema':
            return calc_tema(src, length)
        else:
            return pd.Series(src).rolling(length).mean().values
    
    def hann(src, period):
        result = np.zeros(len(src))
        pi_x2 = 2.0 * np.pi / (period + 1)
        for i in range(period, len(src)):
            sum_coefs = 0.0
            sum_hann = 0.0
            for count in range(1, period + 1):
                coef = 1.0 - np.cos(count * pi_x2)
                sum_coefs += coef
                sum_hann += coef * src[i + count - period]
            result[i] = sum_hann / sum_coefs if sum_coefs != 0 else 0
        return result
    
    def calc_dmh(h, l, period):
        up_move = np.zeros(n)
        dn_move = np.zeros(n)
        up_move[1:] = h[1:] - h[:-1]
        dn_move[1:] = l[:-1] - l[1:]
        p_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
        m_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
        diff = p_dm - m_dm
        hann_result = hann(diff, period)
        dmh_vals = hann_result / period
        return dmh_vals
    
    def calc_cond_ema(x, cond, period):
        result = np.zeros(len(x))
        val = 0.0
        ema_val = None
        for i in range(len(x)):
            if cond[i]:
                if ema_val is None:
                    ema_val = x[i]
                else:
                    ema_val = (x[i] - ema_val) * (2 / (period + 1)) + ema_val
                result[i] = ema_val
            else:
                result[i] = ema_val if ema_val is not None else 0
        return result
    
    def calc_cond_sma(x, cond, period):
        result = np.zeros(len(x))
        vals = []
        for i in range(len(x)):
            if cond[i]:
                vals.append(x[i])
                if len(vals) > period:
                    vals.pop(0)
            result[i] = np.mean(vals) if vals else 0
        return result
    
    def rng_size_calc(x, scale, qty, period):
        tr = calc_tr(high, low, close)
        atr = calc_atr(high, low, close, period)
        prev_x = np.roll(x, 1)
        prev_x[0] = x[0]
        ac = calc_cond_ema(np.abs(x - prev_x), np.ones(n, dtype=bool), period)
        sd = np.sqrt(calc_cond_sma(x**2, np.ones(n, dtype=bool), period) - calc_cond_sma(x, np.ones(n, dtype=bool), period)**2)
        result = np.zeros(n)
        if scale == 'Pips':
            result = qty * 0.0001
        elif scale == 'Points':
            result = qty * 1
        elif scale == '% of Price':
            result = close * qty / 100
        elif scale == 'ATR':
            result = qty * atr
        elif scale == 'Average Change':
            result = qty * ac
        elif scale == 'Standard Deviation':
            result = qty * sd
        elif scale == 'Ticks':
            result = qty * 0.0001
        else:
            result = qty
        return result
    
    def calc_rf(h, l, rng, period, typ, smooth, sn, av_vals, av_n):
        rng_smooth = calc_cond_ema(rng, np.ones(n, dtype=bool), sn)
        r = rng_smooth if smooth else rng
        rfilt = np.zeros(n)
        rfilt[0] = (h[0] + l[0]) / 2
        for i in range(1, n):
            prev_rf = rfilt[i-1]
            if typ == 'Type 1':
                if h[i] - r[i] > prev_rf:
                    rfilt[i] = h[i] - r[i]
                elif l[i] + r[i] < prev_rf:
                    rfilt[i] = l[i] + r[i]
                else:
                    rfilt[i] = prev_rf
            elif typ == 'Type 2':
                if h[i] >= prev_rf + r[i]:
                    rfilt[i] = prev_rf + np.floor(np.abs(h[i] - prev_rf) / r[i]) * r[i]
                elif l[i] <= prev_rf - r[i]:
                    rfilt[i] = prev_rf - np.floor(np.abs(l[i] - prev_rf) / r[i]) * r[i]
                else:
                    rfilt[i] = prev_rf
        prev_rf_shifted = np.roll(rfilt, 1)
        prev_rf_shifted[0] = rfilt[0]
        changed = rfilt != prev_rf_shifted
        if av_vals:
            rf_filt = calc_cond_ema(rfilt, changed, av_n)
            hi_band = calc_cond_ema(rfilt + r, changed, av_n)
            lo_band = calc_cond_ema(rfilt - r, changed, av_n)
        else:
            rf_filt = rfilt
            hi_band = rfilt + r
            lo_band = rfilt - r
        return hi_band, lo_band, rf_filt
    
    rf_avg_price = (high + low) / 2
    rng = rng_size_calc(rf_avg_price, 'Average Change', rfRngQty, rfRngPer)
    rf_h_band, rf_l_band, rf_filt = calc_rf(high, low, rng, rfRngPer, rfFilterType, rfSmoothRange, rfSmoothPer, rfAvVals, rfAvSamples)
    
    rf_fdir = np.zeros(n)
    for i in range(1, n):
        if rf_filt[i] > rf_filt[i-1]:
            rf_fdir[i] = 1
        elif rf_filt[i] < rf_filt[i-1]:
            rf_fdir[i] = -1
        else:
            rf_fdir[i] = rf_fdir[i-1]
    rf_upward = (rf_fdir == 1).astype(int)
    rf_downward = (rf_fdir == -1).astype(int)
    rfLongSignal = (rf_upward == 1) & (close > rf_filt)
    rfShortSignal = (rf_downward == 1) & (close < rf_filt)
    
    def calc_tdfi():
        mma = ma_func(mmaModeTDFI, close * 1000, mmaLengthTDFI)
        smma = ma_func(smmaModeTDFI, mma, smmaLengthTDFI)
        impet_mma = mma - np.roll(mma, 1)
        impet_mma[0] = 0
        impet_smma = smma - np.roll(smma, 1)
        impet_smma[0] = 0
        div_ma = np.abs(mma - smma)
        aver_impet = (impet_mma + impet_smma) / 2
        tdf = np.power(div_ma, 1) * np.power(aver_impet, nLengthTDFI)
        lookback_total = lookbackTDFI * nLengthTDFI
        highest_val = pd.Series(np.abs(tdf)).rolling(lookback_total).max().fillna(1).values
        return tdf / highest_val
    
    signalTDFI = calc_tdfi()
    signalLongTDFI = signalTDFI > filterHighTDFI
    signalShortTDFI = signalTDFI < filterLowTDFI
    finalLongSignalTDFI = signalShortTDFI if inverseTDFI else signalLongTDFI
    finalShortSignalTDFI = signalLongTDFI if inverseTDFI else signalShortTDFI
    
    signalDMH = calc_dmh(high, low, lengthDMH)
    dmhRising = np.zeros(n)
    for i in range(1, n):
        dmhRising[i] = 1 if signalDMH[i] > signalDMH[i-1] else 0
    
    dmhLongSignal = np.zeros(n, dtype=bool)
    for i in range(n):
        if dmhLongCondition == 'Rising':
            dmhLongSignal[i] = dmhRising[i] == 1
        elif dmhLongCondition == 'Above Zero':
            dmhLongSignal[i] = signalDMH[i] > 0
        elif dmhLongCondition == 'Rising & Above Zero':
            dmhLongSignal[i] = (dmhRising[i] == 1) and (signalDMH[i] > 0)
    
    dmhShortSignal = np.zeros(n, dtype=bool)
    for i in range(n):
        if dmhShortCondition == 'Falling':
            dmhShortSignal[i] = dmhRising[i] == 0
        elif dmhShortCondition == 'Below Zero':
            dmhShortSignal[i] = signalDMH[i] < 0
        elif dmhShortCondition == 'Falling & Below Zero':
            dmhShortSignal[i] = (dmhRising[i] == 0) and (signalDMH[i] < 0)
    
    up_move = np.zeros(n)
    dn_move = np.zeros(n)
    up_move[1:] = high[1:] - high[:-1]
    dn_move[1:] = low[:-1] - low[1:]
    pos_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    tr = calc_tr(high, low, close)
    smooth_pos = np.zeros(n)
    smooth_neg = np.zeros(n)
    smooth_tr = np.zeros(n)
    smooth_pos[di_len-1] = np.mean(pos_dm[:di_len])
    smooth_neg[di_len-1] = np.mean(neg_dm[:di_len])
    smooth_tr[di_len-1] = np.mean(tr[:di_len])
    for i in range(di_len, n):
        smooth_pos[i] = (smooth_pos[i-1] * (di_len - 1) + pos_dm[i]) / di_len
        smooth_neg[i] = (smooth_neg[i-1] * (di_len - 1) + neg_dm[i]) / di_len
        smooth_tr[i] = (smooth_tr[i-1] * (di_len - 1) + tr[i]) / di_len
    di_pos = np.where(smooth_tr != 0, smooth_pos / smooth_tr * 100, 0)
    di_neg = np.where(smooth_tr != 0, smooth_neg / smooth_tr * 100, 0)
    dx = np.where((di_pos + di_neg) != 0, np.abs(di_pos - di_neg) / (di_pos + di_neg) * 100, 0)
    adx = np.zeros(n)
    adx[adx_len-1] = np.mean(dx[:adx_len])
    for i in range(adx_len, n):
        adx[i] = (adx[i-1] * (adx_len - 1) + dx[i]) / adx_len
    adx_sma = pd.Series(adx).rolling(adxMALength).mean().fillna(0).values
    
    adxLongFilter = np.zeros(n, dtype=bool)
    adxShortFilter = np.zeros(n, dtype=bool)
    for i in range(n):
        adxLongFilter[i] = (adx[i] >= adxThreshold) and ((not useADXMA) or (adx[i] > adx_sma[i]))
        adxShortFilter[i] = (adx[i] >= adxThreshold) and ((not useADXMA) or (adx[i] > adx_sma[i]))
    
    longSignal = np.zeros(n, dtype=bool)
    shortSignal = np.zeros(n, dtype=bool)
    for i in range(n):
        longSignal[i] = rfLongSignal[i] and finalLongSignalTDFI[i] and dmhLongSignal[i] and adxLongFilter[i]
        shortSignal[i] = rfShortSignal[i] and finalShortSignalTDFI[i] and dmhShortSignal[i] and adxShortFilter[i]
    
    if not requireAllSignals:
        for i in range(n):
            if useTDFI:
                longSignal[i] = longSignal[i] or finalLongSignalTDFI[i]
                shortSignal[i] = shortSignal[i] or finalShortSignalTDFI[i]
            if useDMH:
                longSignal[i] = longSignal[i] or dmhLongSignal[i]
                shortSignal[i] = shortSignal[i] or dmhShortSignal[i]
            if useADX:
                longSignal[i] = longSignal[i] or adxLongFilter[i]
                shortSignal[i] = shortSignal[i] or adxShortFilter[i]
    
    entries = []
    trade_num = 1
    for i in range(n):
        if np.isnan(close[i]):
            continue
        if allowLong and longSignal[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(timestamps[i]),
                'entry_time': datetime.fromtimestamp(timestamps[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': close[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close[i],
                'raw_price_b': close[i]
            })
            trade_num += 1
        elif allowShort and shortSignal[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(timestamps[i]),
                'entry_time': datetime.fromtimestamp(timestamps[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': close[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close[i],
                'raw_price_b': close[i]
            })
            trade_num