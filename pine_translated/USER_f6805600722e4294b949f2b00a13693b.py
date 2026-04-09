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
    results = []
    trade_num = 1

    # Calculate indicators
    high = df['high']
    low = df['low']
    close = df['close']
    open_price = df['open']
    volume = df['volume']

    # SMA for trend filter
    loc = close.rolling(54).mean()

    # ATR (Wilder)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()

    # Volume filter SMA
    vol_sma = volume.rolling(9).mean()

    # Bullish FVG: low > high[2] and close[1] > high[2]
    bull_fvg = (low > high.shift(2)) & (close.shift(1) > high.shift(2))
    # Bearish FVG: high < low[2] and close[1] < low[2]
    bear_fvg = (high < low.shift(2)) & (close.shift(1) < low.shift(2))

    # Filters
    volfilt = volume.shift(1) > vol_sma.shift(1) * 1.5
    atrfilt = ((low - high.shift(2) > atr) | (low.shift(2) - high > atr))
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    # Final conditions with filters
    bull_cond = bull_fvg & volfilt & atrfilt & locfiltb
    bear_cond = bear_fvg & volfilt & atrfilt & locfilts

    # Track mitigated state
    mitigated_bull = pd.Series(False, index=df.index)
    mitigated_bear = pd.Series(False, index=df.index)

    for i in range(2, len(df)):
        if pd.isna(high.iloc[i]) or pd.isna(low.iloc[i]) or pd.isna(close.iloc[i]):
            continue

        # Check if bull FVG at i-2 is now mitigated (price re-enters)
        if i >= 2:
            bull_fvg_past = (low.iloc[i-2] > high.iloc[i-4]) & (close.iloc[i-3] > high.iloc[i-4]) if i >= 4 else False
            if bull_fvg_past and not mitigated_bull.iloc[i-2]:
                if close.iloc[i] >= low.iloc[i-2] and close.iloc[i] <= high.iloc[i-2]:
                    mitigated_bull.iloc[i-2] = True
                    # Long entry signal
                    entry_price = close.iloc[i]
                    entry_ts = df['time'].iloc[i]
                    entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                    results.append({
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

        # Check if bear FVG at i-2 is now mitigated (price re-enters)
        if i >= 2:
            bear_fvg_past = (high.iloc[i-2] < low.iloc[i-4]) & (close.iloc[i-3] < low.iloc[i-4]) if i >= 4 else False
            if bear_fvg_past and not mitigated_bear.iloc[i-2]:
                if close.iloc[i] <= high.iloc[i-2] and close.iloc[i] >= low.iloc[i-2]:
                    mitigated_bear.iloc[i-2] = True
                    # Short entry signal
                    entry_price = close.iloc[i]
                    entry_ts = df['time'].iloc[i]
                    entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                    results.append({
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

        # Also trigger on fresh FVG with strong confirmation
        if bull_cond.iloc[i] and close.iloc[i] > open_price.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            results.append({
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

        if bear_cond.iloc[i] and close.iloc[i] < open_price.iloc[i]:
            entry_price = close.iloc[i]
            entry_ts = df['time'].iloc[i]
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            results.append({
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

    return results