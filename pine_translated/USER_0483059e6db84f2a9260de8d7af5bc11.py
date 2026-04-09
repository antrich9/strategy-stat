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
    # Validate columns
    required = {'time', 'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    # Work on a copy to avoid mutating input
    df = df.copy()

    # ---------- Helper functions ----------
    def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
        """Wilder's RSI implementation."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1.0 / length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def crossover(s1: pd.Series, s2: pd.Series) -> pd.Series:
        """True when s1 crosses above s2."""
        return (s1 > s2) & (s1.shift(1) <= s2.shift(1))

    def crossunder(s1: pd.Series, s2: pd.Series) -> pd.Series:
        """True when s1 crosses below s2."""
        return (s1 < s2) & (s1.shift(1) >= s2.shift(1))

    # ---------- Price series ----------
    high = df['high']
    low = df['low']
    close = df['close']

    # ---------- Double top / bottom shifted values ----------
    high_1 = high.shift(1)
    high_2 = high.shift(2)
    high_3 = high.shift(3)
    high_4 = high.shift(4)

    low_1 = low.shift(1)
    low_2 = low.shift(2)
    low_3 = low.shift(3)
    low_4 = low.shift(4)

    # ---------- Double top detection ----------
    doubleTop = (
        (high_4 < high_3) &
        (high_3 < high_2) &
        (high_2 > high_1) &
        (high_1 < high) &
        (high_2 > high) &
        (np.abs(high_2 - high) <= high * 0.01)
    )

    # ---------- Double bottom detection ----------
    doubleBottom = (
        (low_4 > low_3) &
        (low_3 > low_2) &
        (low_2 < low_1) &
        (low_1 > low) &
        (low_2 < low) &
        (np.abs(low_2 - low) <= low * 0.01)
    )

    # ---------- Retracement levels ----------
    retracementLevel = 0.618  # default from script

    retracementTop = np.where(doubleTop, high - (high - low_2) * retracementLevel, np.nan)
    retracementBottom = np.where(doubleBottom, low + (high_2 - low) * retracementLevel, np.nan)

    retracementTop = pd.Series(retracementTop, index=df.index)
    retracementBottom = pd.Series(retracementBottom, index=df.index)

    # ---------- Signals ----------
    longSignal = (crossover(close, retracementBottom) & doubleBottom).fillna(False)
    shortSignal = (crossunder(close, retracementTop) & doubleTop).fillna(False)

    # ---------- RSI ----------
    rsi = compute_rsi(close, length=14)

    # ---------- Divergence conditions ----------
    rsiDivergenceBullish = (doubleBottom & (rsi.shift(2) > rsi) & (rsi < 30)).fillna(False)
    rsiDivergenceBearish = (doubleTop & (rsi.shift(2) > rsi) & (rsi > 70)).fillna(False)

    # ---------- Final entry conditions ----------
    long_entry = (longSignal & rsiDivergenceBullish).fillna(False)
    short_entry = (shortSignal & rsiDivergenceBearish).fillna(False)

    # ---------- Generate entry list ----------
    entries = []
    trade_num = 1

    for i in df.index:
        if long_entry.loc[i]:
            entry_ts = int(df.loc[i, 'time'])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df.loc[i, 'close'])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_entry.loc[i]:
            entry_ts = int(df.loc[i, 'time'])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df.loc[i, 'close'])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries