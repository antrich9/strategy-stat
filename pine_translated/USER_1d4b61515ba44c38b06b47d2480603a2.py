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
    inp1 = False
    inp2 = False
    inp3 = False

    def wilder_rsi(series: pd.Series, length: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        return atr

    atr = wilder_atr(df['high'], df['low'], df['close'], 20) / 1.5
    vol_sma = df['volume'].rolling(9).mean()
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)

    volfilt = inp1 | (df['volume'].shift(1) > vol_sma * 1.5)
    atrfilt = inp2 | ((df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr))
    locfiltb = inp3 | loc2
    locfilts = inp3 | (~loc2)

    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts

    dt_series = pd.to_datetime(df['time'], unit='s', utc=True)
    london_dt = dt_series.dt.tz_convert('Europe/London')
    hour = london_dt.dt.hour
    minute = london_dt.dt.minute

    is_within_time_window = (
        ((hour == 7) & (minute >= 45)) |
        ((hour == 8) & (minute <= 45)) |
        ((hour == 9) & (minute <= 45) & (hour < 10)) |
        ((hour == 14) & (minute >= 45)) |
        ((hour == 15) & (minute <= 45)) |
        ((hour == 16) & (minute <= 45) & (hour < 17))
    )

    entries = []
    trade_num = 0

    for i in range(len(df)):
        if pd.isna(bfvg.iloc[i]) or pd.isna(sfvg.iloc[i]) or pd.isna(is_within_time_window.iloc[i]):
            continue
        if pd.isna(df['low'].iloc[i]) or pd.isna(df['close'].iloc[i]):
            continue

        if bfvg.iloc[i] and is_within_time_window.iloc[i]:
            trade_num += 1
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })

        if sfvg.iloc[i] and is_within_time_window.iloc[i]:
            trade_num += 1
            ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(ts),
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })

    return entries