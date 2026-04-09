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
    # Parameters (matching Pine script inputs)
    baseline_length = 50
    min_body_pct = 70.0
    cmf_length = 20

    # Baseline EMA
    baseline = df['close'].ewm(span=baseline_length, adjust=False).mean()

    # Elephant candle detection
    body = (df['close'] - df['open']).abs()
    full = df['high'] - df['low']
    elephant = body >= (min_body_pct / 100.0) * full
    is_green = (df['close'] > df['open']) & elephant
    is_red = (df['close'] < df['open']) & elephant

    # Chaikin Money Flow
    high_low_diff = df['high'] - df['low']
    mfm = np.where(high_low_diff != 0,
                   (2 * df['close'] - df['high'] - df['low']) / high_low_diff,
                   np.nan)
    mfv_series = pd.Series(mfm * df['volume'].values, index=df.index)
    sum_mfv = mfv_series.rolling(window=cmf_length).sum()
    sum_vol = df['volume'].rolling(window=cmf_length).sum()
    cmf = sum_mfv / sum_vol

    # Entry condition series
    long_cond = (df['close'] > baseline) & is_green & (cmf > 0)
    short_cond = (df['close'] < baseline) & is_red & (cmf < 0)

    entries = []
    trade_num = 1

    for i in df.index:
        # Skip bars where required indicators are NaN
        if pd.isna(baseline.iloc[i]) or pd.isna(cmf.iloc[i]):
            continue
        if long_cond.iloc[i]:
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
        elif short_cond.iloc[i]:
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