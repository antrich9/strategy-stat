import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    
    # E2PSS Filter
    PeriodE2PSS = 15
    pi = 2 * np.arcsin(1)
    
    a1 = np.exp(-1.414 * pi / PeriodE2PSS)
    b1 = 2 * a1 * np.cos(1.414 * pi / PeriodE2PSS)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    PriceE2PSS = (high + low) / 2
    Filt2 = np.zeros(len(df))
    TriggerE2PSS = np.zeros(len(df))
    
    for i in range(len(df)):
        if i < 3:
            Filt2[i] = PriceE2PSS.iloc[i]
        else:
            Filt2[i] = coef1 * PriceE2PSS.iloc[i] + coef2 * Filt2[i-1] + coef3 * Filt2[i-2]
        TriggerE2PSS[i] = Filt2[i-1] if i >= 1 else PriceE2PSS.iloc[i]
    
    Filt2 = pd.Series(Filt2, index=df.index)
    TriggerE2PSS = pd.Series(TriggerE2PSS, index=df.index)
    
    signalLongE2PSS = Filt2 > TriggerE2PSS
    signalShortE2PSS = Filt2 < TriggerE2PSS
    
    # Trendilo - ALMA implementation
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    pct_change = close.diff(trendilo_smooth) / close * 100
    
    def alma_rolling(series, length, offset, sigma):
        result = pd.Series(np.nan, index=series.index)
        for i in range(length - 1, len(series)):
            window = series.iloc[i - length + 1:i + 1].values
            k = np.arange(length)
            w = np.exp(-0.5 * ((k - offset * (length - 1)) / sigma) ** 2)
            w = w / w.sum()
            result.iloc[i] = np.sum(window * w)
        return result
    
    avg_pct_change = alma_rolling(pct_change, trendilo_length, trendilo_offset, trendilo_sigma)
    rms = trendilo_bmult * np.sqrt((avg_pct_change ** 2).rolling(trendilo_length).mean())
    trendilo_dir = pd.Series(np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0)), index=df.index)
    
    # TTMS - Bollinger Bands
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    dev_TTMS = BB_mult_TTMS * close.rolling(length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + dev_TTMS
    BB_lower_TTMS = BB_basis_TTMS - dev_TTMS
    
    # TTMS - Keltner Channels
    KC_mult_high_TTMS = 1.0
    KC_mult_mid_TTMS = 1.5
    KC_mult_low_TTMS = 2.0
    
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = pd.Series(np.maximum(np.maximum(tr1, tr2), tr3), index=df.index)
    devKC_TTMS = tr.rolling(length_TTMS).mean()
    
    KC_upper_high_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_high_TTMS
    KC_lower_high_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_high_TTMS
    KC_upper_mid_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_mid_TTMS
    KC_lower_mid_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_mid_TTMS
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    
    # TTMS - Squeeze conditions
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    LowSqz_TTMS = (BB_lower_TTMS >= KC_lower_low_TTMS) | (BB_upper_TTMS <= KC_upper_low_TTMS)
    MidSqz_TTMS = (BB_lower_TTMS >= KC_lower_mid_TTMS) | (BB_upper_TTMS <= KC_upper_mid_TTMS)
    HighSqz_TTMS = (BB_lower_TTMS >= KC_lower_high_TTMS) | (BB_upper_TTMS <= KC_upper_high_TTMS)
    
    # TTMS - Momentum (linreg)
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    avg_val = (highest_high + lowest_low + BB_basis_TTMS) / 3
    mom_TTMS = (close - avg_val).rolling(length_TTMS).mean()
    
    prev_mom = mom_TTMS.shift(1)
    TTMS_Signals_TTMS = pd.Series(np.where(mom_TTMS > 0, np.where(mom_TTMS > prev_mom, 1, 2), np.where(mom_TTMS < prev_mom, -1, -2)), index=df.index)
    
    # TTMS - Basic conditions
    basicLongCondition_TTMS = TTMS_Signals_TTMS > 0
    basicShortCondition_TTMS = TTMS_Signals_TTMS < 0
    
    # TTMS - Final conditions with highlight and cross
    TTMS_SignalsLong_TTMS = NoSqz_TTMS & basicLongCondition_TTMS
    TTMS_SignalsShort_TTMS = NoSqz_TTMS & basicShortCondition_TTMS
    
    TTMS_SignalsLongCross_TTMS = ~TTMS_SignalsLong_TTMS.shift(1).fillna(False) & TTMS_SignalsLong_TTMS
    TTMS_SignalsShortCross_TTMS = ~TTMS_SignalsShort_TTMS.shift(1).fillna(False) & TTMS_SignalsShort_TTMS
    
    TTMS_SignalsLongFinal_TTMS = TTMS_SignalsLongCross_TTMS
    TTMS_SignalsShortFinal_TTMS = TTMS_SignalsShortCross_TTMS
    
    # Entry conditions
    long_condition = signalLongE2PSS & (trendilo_dir == 1) & TTMS_SignalsLongFinal_TTMS
    short_condition = signalShortE2PSS & (trendilo_dir == -1) & TTMS_SignalsShortFinal_TTMS
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(2, len(df)):
        if long_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': close.iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': close.iloc[i],
                'raw_price_b': close.iloc[i]
            })
            trade_num += 1
    
    return entries