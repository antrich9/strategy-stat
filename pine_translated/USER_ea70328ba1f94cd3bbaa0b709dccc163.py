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
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)

    # H4 High/Low calculation using M15 data resampled to H4
    df_h4 = df.set_index('datetime').resample('4H').agg({'high': 'max', 'low': 'min'})
    h4_high = df_h4['high'].reindex(df['datetime'], method='ffill')
    h4_low = df_h4['low'].reindex(df['datetime'], method='ffill')

    # Liquidity sweep detection
    liquidity_sweep_high = (df['close'] >= h4_high * 0.995) & (df['close'] <= h4_high * 1.005)
    liquidity_sweep_low = (df['close'] <= h4_low * 1.005) & (df['close'] >= h4_low * 0.995)
    h4_confirmation = liquidity_sweep_high | liquidity_sweep_low

    # Time windows based on hour
    df['h'] = df['datetime'].dt.hour
    pre_london = df['h'] == 7
    london_open = (df['h'] >= 8) & (df['h'] < 10)
    post_london = (df['h'] >= 10) & (df['h'] < 12)
    london_ny_overlap = (df['h'] >= 13) & (df['h'] < 16)
    late_overlap = (df['h'] >= 16) & (df['h'] < 17)

    # OB + FVG detection
    is_up = df['close'] > df['open']
    is_down = df['close'] < df['open']
    ob_up = is_down.shift(1) & is_up & (df['close'] > df['high'].shift(1))
    ob_down = is_up.shift(1) & is_down & (df['close'] < df['low'].shift(1))
    fvg_up = df['low'] > df['high'].shift(2)
    fvg_down = df['high'] < df['low'].shift(2)
    stacked_bullish = ob_up & fvg_up
    stacked_bearish = ob_down & fvg_down

    # Track open trades to prevent entries when already in a trade
    in_trade = pd.Series([False] * len(df))

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i < 2:
            continue

        if stacked_bullish.iloc[i] and london_open.iloc[i] and h4_confirmation.iloc[i] and not in_trade.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
            in_trade.iloc[i] = True
            continue

        if stacked_bearish.iloc[i] and london_open.iloc[i] and h4_confirmation.iloc[i] and not in_trade.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
            in_trade.iloc[i] = True
            continue

        if stacked_bullish.iloc[i] and post_london.iloc[i] and not in_trade.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
            in_trade.iloc[i] = True
            continue

        if stacked_bearish.iloc[i] and post_london.iloc[i] and not in_trade.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
            in_trade.iloc[i] = True
            continue

        if stacked_bullish.iloc[i] and london_ny_overlap.iloc[i] and h4_confirmation.iloc[i] and not in_trade.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
            in_trade.iloc[i] = True
            continue

        if stacked_bearish.iloc[i] and london_ny_overlap.iloc[i] and h4_confirmation.iloc[i] and not in_trade.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
            in_trade.iloc[i] = True
            continue

        if stacked_bullish.iloc[i] and late_overlap.iloc[i] and h4_confirmation.iloc[i] and not in_trade.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
            in_trade.iloc[i] = True
            continue

        if stacked_bearish.iloc[i] and late_overlap.iloc[i] and h4_confirmation.iloc[i] and not in_trade.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
            in_trade.iloc[i] = True

    return entries