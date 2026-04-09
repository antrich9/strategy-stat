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
    pp = 5
    atr_length = 14
    atr_multiplier = 1.5
    tp_multiplier = 2.0

    def wilder_rsi(src, length):
        delta = src.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
        return atr

    atr_val = wilder_atr(df['high'], df['low'], df['close'], atr_length)

    def pivot_high(src, length):
        ph = src.rolling(window=2*length+1, center=True).max()
        check = pd.concat([src.shift(length), ph], axis=1)
        return check.max(axis=1) == src.shift(length)

    def pivot_low(src, length):
        pl = src.rolling(window=2*length+1, center=True).min()
        check = pd.concat([src.shift(length), pl], axis=1)
        return check.min(axis=1) == src.shift(length)

    high_pivot = pivot_high(df['high'], pp)
    low_pivot = pivot_low(df['low'], pp)

    high_pivot_shifted = high_pivot.shift(pp)
    low_pivot_shifted = low_pivot.shift(pp)

    dt_trade = False
    db_trade = False
    dbTradeTriggered = False
    dtTradeTriggered = False

    entries = []
    trade_num = 1

    for i in range(2*pp + 10, len(df)):
        if pd.isna(atr_val.iloc[i]):
            continue

        if high_pivot_shifted.iloc[i]:
            dt_trade = True
            db_trade = False
        elif low_pivot_shifted.iloc[i]:
            db_trade = True
            dt_trade = False

        atr_curr = atr_val.iloc[i]
        stop_distance = atr_curr * atr_multiplier
        tp_distance = atr_curr * tp_multiplier

        if dt_trade:
            dbTradeTriggered = False
            dtTradeTriggered = True
        elif db_trade:
            dtTradeTriggered = False
            dbTradeTriggered = True

        if dtTradeTriggered and atr_curr > 0:
            entry_price = df['close'].iloc[i]
            direction = 'long'
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            dtTradeTriggered = False
            dt_trade = False

        if dbTradeTriggered and atr_curr > 0:
            entry_price = df['close'].iloc[i]
            direction = 'short'
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            dbTradeTriggered = False
            db_trade = False

    return entries