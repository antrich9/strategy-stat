import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']
    timestamps = df['time']
    
    n = len(close)
    
    # E2PSS parameters
    PeriodE2PSS = 15
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    Filt2 = np.zeros(n)
    TriggerE2PSS = np.zeros(n)
    
    for i in range(n):
        if i < 2:
            Filt2[i] = (high.iloc[i] + low.iloc[i]) / 2
        else:
            Filt2[i] = coef1 * (high.iloc[i] + low.iloc[i]) / 2 + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
        TriggerE2PSS[i] = Filt2[i-1] if i > 0 else 0
    
    signalLongE2PSS_series = pd.Series(Filt2 > TriggerE2PSS, index=df.index)
    signalShortE2PSS_series = pd.Series(Filt2 < TriggerE2PSS, index=df.index)
    
    # Trendilo
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    pct_change = close.diff(trendilo_smooth) / close * 100
    
    # ALMA approximation using ewm (Wilder's smoothing variant with ALMA offset/sigma)
    def alma_approx(series, length, offset, sigma):
        w = np.exp(-np.arange(length)**2 / (2 * sigma**2))
        w = w / w.sum()
        return series.rolling(length).apply(lambda x: np.dot(x, w[::-1]), raw=True)
    
    avg_pct_change = alma_approx(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    
    rms = trendilo_bmult * avg_pct_change.rolling(trendilo_length).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
    trendilo_dir = pd.Series(np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0)), index=df.index)
    
    # TTM Squeeze
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    BB_basis_TTMS = close.rolling(window=length_TTMS).mean()
    dev_TTMS = BB_mult_TTMS * close.rolling(window=length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS
    
    KC_mult_low_TTMS = 2.0
    KC_basis_TTMS = close.rolling(window=length_TTMS).mean()
    devKC_TTMS = close.rolling(window=length_TTMS).mean()
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    
    # TTM Momentum
    highest_high = high.rolling(window=length_TTMS).max()
    lowest_low = low.rolling(window=length_TTMS).min()
    avg_high_low = (highest_high + lowest_low) / 2
    avg_close_sma = close.rolling(window=length_TTMS).mean()
    mom_source = close - avg_high_low - avg_close_sma
    
    mom_TTMS = mom_source.rolling(window=length_TTMS).apply(lambda x: np.polyfit(np.arange(len(x)), x, 0)[0] if len(x) == length_TTMS else np.nan, raw=True)
    
    TTMS_Signals_TTMS = pd.Series(np.where(mom_TTMS > 0, 1, -1), index=df.index)
    
    basicLongCondition_TTMS = pd.Series(np.where(mom_TTMS > 0, 1, -1), index=df.index)
    basicShortCondition_TTMS = pd.Series(np.where(mom_TTMS < 0, -1, -2), index=df.index)
    
    TTMS_SignalsLong_TTMS = NoSqz_TTMS & (basicLongCondition_TTMS == 1)
    TTMS_SignalsShort_TTMS = NoSqz_TTMS & (basicShortCondition_TTMS == -1)
    
    TTMS_SignalsLongCross_TTMS = (~TTMS_SignalsLong_TTMS.shift(1).fillna(False)) & TTMS_SignalsLong_TTMS
    TTMS_SignalsShortCross_TTMS = (~TTMS_SignalsShort_TTMS.shift(1).fillna(False)) & TTMS_SignalsShort_TTMS
    
    TTMS_SignalsLongFinal_TTMS = TTMS_SignalsLongCross_TTMS
    TTMS_SignalsShortFinal_TTMS = TTMS_SignalsShortCross_TTMS
    
    # Trend and volume
    sma50 = close.rolling(window=50).mean()
    sma200 = close.rolling(window=200).mean()
    uptrend = sma50 > sma200
    downtrend = sma50 < sma200
    
    volMAPeriod = 20
    volMultiplier = 1.5
    volMA = volume.rolling(window=volMAPeriod).mean()
    volumeSurge = volume > volMA * volMultiplier
    volConfirmed = volumeSurge
    
    trendValid = uptrend | downtrend
    
    sma20 = close.rolling(window=20).mean()
    
    maxTradesPerDay = 2
    tradesToday = np.zeros(n)
    tradedThisBar = np.zeros(n, dtype=bool)
    
    long_condition = pd.Series(False, index=df.index)
    short_condition = pd.Series(False, index=df.index)
    
    trades = []
    trade_num = 1
    
    for i in range(1, n):
        if i > 0 and pd.Timestamp(timestamps.iloc[i], unit='ms').date() != pd.Timestamp(timestamps.iloc[i-1], unit='ms').date():
            tradesToday[i] = 0
        else:
            tradesToday[i] = tradesToday[i-1] if i > 0 else 0
        
        tradedThisBar[i] = tradedThisBar[i-1] if i > 0 else False
        
        tradeAllowed = (tradesToday[i] < maxTradesPerDay) and not tradedThisBar[i]
        
        longTriggered = uptrend.iloc[i] and volConfirmed.iloc[i] and trendValid.iloc[i] and (close.iloc[i] > sma20.iloc[i]) and tradeAllowed
        shortTriggered = downtrend.iloc[i] and volConfirmed.iloc[i] and trendValid.iloc[i] and (close.iloc[i] < sma20.iloc[i]) and tradeAllowed
        
        long_condition.iloc[i] = longTriggered and signalLongE2PSS_series.iloc[i] and (trendilo_dir.iloc[i] == 1) and (basicLongCondition_TTMS.iloc[i] > 0)
        short_condition.iloc[i] = shortTriggered and signalShortE2PSS_series.iloc[i] and (trendilo_dir.iloc[i] == -1) and (basicShortCondition_TTMS.iloc[i] < 0)
        
        if long_condition.iloc[i]:
            tradesToday[i] += 1
            tradedThisBar[i] = True
            trades.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(timestamps.iloc[i]),
                'entry_time': datetime.fromtimestamp(timestamps.iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            tradesToday[i] += 1
            tradedThisBar[i] = True
            trades.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(timestamps.iloc[i]),
                'entry_time': datetime.fromtimestamp(timestamps.iloc[i] / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
    
    return trades