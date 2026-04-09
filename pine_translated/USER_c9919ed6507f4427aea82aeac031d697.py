import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.
    """
    length = 21
    offset = 0
    length2 = 6
    rsiLength = 14

    src = df['close'].copy()

    def calc_wilder_ema(series, length):
        alpha = 1.0 / length
        return series.ewm(alpha=alpha, adjust=False).mean()

    delta = src.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = calc_wilder_ema(gain, rsiLength)
    avg_loss = calc_wilder_ema(loss, rsiLength)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    def rolling_linreg(y, length):
        result = pd.Series(np.nan, index=y.index, dtype=np.float64)
        for i in range(length - 1, len(y)):
            window = y.iloc[i - length + 1:i + 1]
            x = np.arange(length)
            if not np.any(np.isnan(window)):
                coeffs = np.polyfit(x, window.values, 1)
                slope, intercept = coeffs[0], coeffs[1]
                result.iloc[i] = intercept + slope * (length - 1 - offset)
        return result

    lsma = rolling_linreg(src, length)
    lsma2 = rolling_linreg(lsma, length)
    b = lsma - lsma2
    zlsma2 = lsma + b
    trig2 = zlsma2.rolling(length2).mean()

    crossover = (zlsma2 > trig2) & (zlsma2.shift(1) <= trig2.shift(1))
    crossunder = (zlsma2 < trig2) & (zlsma2.shift(1) >= trig2.shift(1))

    longCondition = crossover & (rsi > 50)
    shortCondition = crossunder & (rsi < 50)

    entries = []
    trade_num = 1

    for i in longCondition.index:
        if pd.isna(longCondition.iloc[i]) or pd.isna(src.iloc[i]):
            continue
        if longCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entry_price_guess = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1

    for i in shortCondition.index:
        if pd.isna(shortCondition.iloc[i]) or pd.isna(src.iloc[i]):
            continue
        if shortCondition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            entry_price_guess = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price_guess,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price_guess,
                'raw_price_b': entry_price_guess
            })
            trade_num += 1

    return entries