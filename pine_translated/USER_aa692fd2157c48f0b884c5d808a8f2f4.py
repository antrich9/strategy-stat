import pandas as pd
import numpy as np
from datetime import datetime, timezone

def alma(data, length, offset, sigma):
    m = offset * (length - 1)
    s = sigma * length / 6
    w = np.exp(-(np.arange(length) - m)**2 / (2 * s**2))
    result = np.full(len(data), np.nan)
    if len(data) >= length:
        conv = np.convolve(data, w / w.sum(), mode='valid')
        result[length - 1:] = conv
    return pd.Series(result)

def linear_regression(series, length):
    result = pd.Series(np.nan, index=series.index)
    for i in range(length - 1, len(series)):
        y = series.iloc[i - length + 1:i + 1].values
        x = np.arange(length)
        x_mean = x.mean()
        y_mean = y.mean()
        cov_xy = np.sum((x - x_mean) * (y - y_mean))
        var_x = np.sum((x - x_mean) ** 2)
        if var_x != 0:
            slope = cov_xy / var_x
            intercept = y_mean - slope * x_mean
            result.iloc[i] = slope * (length - 1) + intercept
    return result

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']
    open_col = df['open']
    volume = df['volume']

    # Trendilo
    pct_change = close.diff() / close * 100
    avg_pct_change = alma(pct_change, 50, 0.85, 6)
    squared = avg_pct_change ** 2
    rms = np.sqrt(squared.rolling(50).sum() / 50)
    trendilo_dir = pd.Series(np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0)), index=df.index)

    # E2PSS
    PeriodE2PSS = 15
    a1_val = np.exp(-1.414 * 3.14159 / PeriodE2PSS)
    b1_val = 2 * a1_val * np.cos(1.414 * 3.14159 / PeriodE2PSS)
    coef2 = b1_val
    coef3 = -a1_val * a1_val
    coef1_val = 1 - coef2 - coef3

    Filt2 = pd.Series(0.0, index=df.index)
    TriggerE2PSS = pd.Series(0.0, index=df.index)
    price_e2pss = (high + low) / 2

    for i in range(len(df)):
        if i < 2:
            Filt2.iloc[i] = price_e2pss.iloc[i]
        else:
            prev1 = Filt2.iloc[i - 1] if not pd.isna(Filt2.iloc[i - 1]) else price_e2pss.iloc[i - 1]
            prev2 = Filt2.iloc[i - 2] if not pd.isna(Filt2.iloc[i - 2]) else price_e2pss.iloc[i - 2]
            Filt2.iloc[i] = coef1_val * price_e2pss.iloc[i] + coef2 * prev1 + coef3 * prev2
        TriggerE2PSS.iloc[i] = Filt2.iloc[i - 1] if i > 0 else price_e2pss.iloc[i]

    signalLongE2PSS = Filt2 > TriggerE2PSS
    signalShortE2PSS = Filt2 < TriggerE2PSS

    # TTM Squeeze
    length_TTMS = 20
    BB_basis = close.rolling(length_TTMS).mean()
    dev_BB = close.rolling(length_TTMS).std()
    BB_upper = BB_basis + 2.0 * dev_BB
    BB_lower = BB_basis - 2.0 * dev_BB

    KC_basis = close.rolling(length_TTMS).mean()
    tr = pd.concat([high - low, (high - close.shift()).abs(), (close.shift() - low).abs()], axis=1).max(axis=1)
    devKC = tr.rolling(length_TTMS).mean()
    KC_upper_low = KC_basis + devKC * 2.0
    KC_lower_low = KC_basis - devKC * 2.0

    NoSqz = (BB_lower < KC_lower_low) | (BB_upper > KC_upper_low)

    highest_high = high.rolling(length_TTMS).max()
    lowest_low = low.rolling(length_TTMS).min()
    mom_raw = close - ((highest_high + lowest_low) / 2 + close.rolling(length_TTMS).mean()) / 2
    mom_TTMS = linear_regression(mom_raw, length_TTMS)
    TTMS_Signals_TTMS = pd.Series(np.where(mom_TTMS > 0, 1, -1), index=df.index)

    basicLongCond = TTMS_Signals_TTMS == 1
    basicShortCond = TTMS_Signals_TTMS == -1

    TTMS_Long = NoSqz & basicLongCond
    TTMS_Short = NoSqz & basicShortCond

    TTMS_Long_Cross = (~TTMS_Long.shift(1).fillna(False)) & TTMS_Long
    TTMS_Short_Cross = (~TTMS_Short.shift(1).fillna(False)) & TTMS_Short

    long_condition = signalLongE2PSS & (trendilo_dir == 1) & basicLongCond & TTMS_Long
    short_condition = signalShortE2PSS & (trendilo_dir == -1) & basicShortCond & TTMS_Short

    entries = []
    trade_num = 1

    for i in range(1, len(df)):
        if pd.isna(Filt2.iloc[i]) or pd.isna(avg_pct_change.iloc[i]) or pd.isna(mom_TTMS.iloc[i]):
            continue
        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1

    return entries