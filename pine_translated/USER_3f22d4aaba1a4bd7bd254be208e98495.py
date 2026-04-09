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
    
    lengthjmaJMA = 7
    phasejmaJMA = 50
    powerjmaJMA = 2
    usejmaJMA = True
    usecolorjmaJMA = True
    inverseJMA = True
    highlightMovementsjmaJMA = True
    
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    use_TTMS = True
    redGreen_TTMS = True
    cross_TTMS = True
    inverse_TTMS = False
    highlightMovements_TTMS = True
    length_TTMS = 20
    BB_mult_TTMS = 2.0
    KC_mult_high_TTMS = 1.0
    KC_mult_mid_TTMS = 1.5
    KC_mult_low_TTMS = 2.0
    
    phasejmaJMARatiojmaJMA = 0.5 if phasejmaJMA < -100 else (2.5 if phasejmaJMA > 100 else phasejmaJMA / 100 + 1.5)
    betajmaJMA = 0.45 * (lengthjmaJMA - 1) / (0.45 * (lengthjmaJMA - 1) + 2)
    alphajmaJMA = betajmaJMA ** powerjmaJMA
    
    jmaJMA = pd.Series(index=df.index, dtype=float)
    e0JMA = pd.Series(0.0, index=df.index)
    e1JMA = pd.Series(0.0, index=df.index)
    e2JMA = pd.Series(0.0, index=df.index)
    
    srcjmaJMA = close
    for i in range(len(df)):
        if i == 0:
            e0JMA.iloc[i] = (1 - alphajmaJMA) * srcjmaJMA.iloc[i]
        else:
            e0JMA.iloc[i] = (1 - alphajmaJMA) * srcjmaJMA.iloc[i] + alphajmaJMA * e0JMA.iloc[i-1]
        
        if i == 0:
            e1JMA.iloc[i] = (srcjmaJMA.iloc[i] - e0JMA.iloc[i]) * (1 - betajmaJMA)
        else:
            e1JMA.iloc[i] = (srcjmaJMA.iloc[i] - e0JMA.iloc[i]) * (1 - betajmaJMA) + betajmaJMA * e1JMA.iloc[i-1]
        
        prev_jma = jmaJMA.iloc[i-1] if i > 0 else 0.0
        prev_e2 = e2JMA.iloc[i-1] if i > 0 else 0.0
        e2JMA.iloc[i] = (e0JMA.iloc[i] + phasejmaJMARatiojmaJMA * e1JMA.iloc[i] - prev_jma) * (1 - alphajmaJMA) ** 2 + alphajmaJMA ** 2 * prev_e2
        
        if i == 0:
            jmaJMA.iloc[i] = e2JMA.iloc[i]
        else:
            jmaJMA.iloc[i] = e2JMA.iloc[i] + jmaJMA.iloc[i-1]
    
    signalmaJMALong_raw = close > jmaJMA if not usecolorjmaJMA else ((jmaJMA > jmaJMA.shift(1)) & (close > jmaJMA))
    signalmaJMAShort_raw = close < jmaJMA if not usecolorjmaJMA else ((jmaJMA < jmaJMA.shift(1)) & (close < jmaJMA))
    
    signalmaJMALong = signalmaJMALong_raw if not usejmaJMA else signalmaJMALong_raw
    signalmaJMAShort = signalmaJMAShort_raw if not usejmaJMA else signalmaJMAShort_raw
    
    finalLongSignalJMA = signalmaJMAShort if inverseJMA else signalmaJMALong
    finalShortSignalJMA = signalmaJMALong if inverseJMA else signalmaJMAShort
    
    pct_change = close.diff(trendilo_smooth) / close * 100
    avg_pct_change = pct_change.ewm(span=trendilo_length, adjust=False).mean()
    
    rms_vals = pd.Series(index=df.index, dtype=float)
    for i in range(trendilo_length - 1, len(df)):
        window = avg_pct_change.iloc[i - trendilo_length + 1:i + 1]
        rms_vals.iloc[i] = trendilo_bmult * np.sqrt((window ** 2).mean())
    
    trendilo_dir = pd.Series(0, index=df.index, dtype=int)
    trendilo_dir = trendilo_dir.where(~(avg_pct_change > rms_vals), 1)
    trendilo_dir = trendilo_dir.where(~(avg_pct_change < -rms_vals), -1)
    trendilo_dir = trendilo_dir.fillna(0).astype(int)
    
    BB_basis_TTMS = close.rolling(length_TTMS).mean()
    BB_std = close.rolling(length_TTMS).std()
    BB_upper_TTMS = BB_basis_TTMS + BB_mult_TTMS * BB_std
    BB_lower_TTMS = BB_basis_TTMS - BB_mult_TTMS * BB_std
    
    KC_basis_TTMS = close.rolling(length_TTMS).mean()
    tr = pd.concat([high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1))], axis=1).max(axis=1)
    devKC_TTMS = tr.rolling(length_TTMS).mean()
    KC_upper_high_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_high_TTMS
    KC_lower_high_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_high_TTMS
    KC_upper_mid_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_mid_TTMS
    KC_lower_mid_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_mid_TTMS
    KC_upper_low_TTMS = KC_basis_TTMS + devKC_TTMS * KC_mult_low_TTMS
    KC_lower_low_TTMS = KC_basis_TTMS - devKC_TTMS * KC_mult_low_TTMS
    
    NoSqz_TTMS = (BB_lower_TTMS < KC_lower_low_TTMS) | (BB_upper_TTMS > KC_upper_low_TTMS)
    LowSqz_TTMS = (BB_lower_TTMS >= KC_lower_low_TTMS) | (BB_upper_TTMS <= KC_upper_low_TTMS)
    MidSqz_TTMS = (BB_lower_TTMS >= KC_lower_mid_TTMS) | (BB_upper_TTMS <= KC_upper_mid_TTMS)
    HighSqz_TTMS = (BB_lower_TTMS >= KC_lower_high_TTMS) | (BB_upper_TTMS <= KC_upper_high_TTMS)
    
    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    avg_val = ((highest_high + lowest_low) / 2 + close.rolling(length_TTMS).mean()) / 2
    mom_TTMS = (close - avg_val).rolling(length_TTMS).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] * (length_TTMS - 1) / 2 + x[0], raw=True)
    mom_TTMS = pd.Series(mom_TTMS, index=df.index)
    
    TTMS_Signals_TTMS = pd.Series(1, index=df.index)
    TTMS_Signals_TTMS = TTMS_Signals_TTMS.where(mom_TTMS > mom_TTMS.shift(1).fillna(0), -1)
    TTMS_Signals_TTMS = TTMS_Signals_TTMS.where(mom_TTMS < mom_TTMS.shift(1).fillna(0), 2)
    TTMS_Signals_TTMS = TTMS_Signals_TTMS.where(mom_TTMS > mom_TTMS.shift(1).fillna(0), -2)
    
    basicLongCondition_TTMS = (TTMS_Signals_TTMS == 1) if redGreen_TTMS else (TTMS_Signals_TTMS > 0)
    basicShortCondition_TTMS = (TTMS_Signals_TTMS == -1) if redGreen_TTMS else (TTMS_Signals_TTMS < 0)
    
    TTMS_SignalsLong_TTMS = basicLongCondition_TTMS if not highlightMovements_TTMS else (NoSqz_TTMS & basicLongCondition_TTMS)
    TTMS_SignalsShort_TTMS = basicShortCondition_TTMS if not highlightMovements_TTMS else (NoSqz_TTMS & basicShortCondition_TTMS)
    
    TTMS_SignalsLongCross_TTMS = TTMS_SignalsLong_TTMS if not cross_TTMS else (TTMS_SignalsLong_TTMS & ~TTMS_SignalsLong_TTMS.shift(1).fillna(False))
    TTMS_SignalsShortCross_TTMS = TTMS_SignalsShort_TTMS if not cross_TTMS else (TTMS_SignalsShort_TTMS & ~TTMS_SignalsShort_TTMS.shift(1).fillna(False))
    
    TTMS_SignalsLongFinal_TTMS = TTMS_SignalsLongCross_TTMS if not inverse_TTMS else TTMS_SignalsShortCross_TTMS
    TTMS_SignalsShortFinal_TTMS = TTMS_SignalsShortCross_TTMS if not inverse_TTMS else TTMS_SignalsLongCross_TTMS
    
    long_condition = finalLongSignalJMA & (trendilo_dir == 1) & basicLongCondition_TTMS
    short_condition = finalShortSignalJMA & (trendilo_dir == -1) & basicShortCondition_TTMS
    
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if np.isnan(jmaJMA.iloc[i]) or np.isnan(avg_pct_change.iloc[i]) or np.isnan(mom_TTMS.iloc[i]):
            continue
        
        if long_condition.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
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
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries