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
    entries = []
    trade_num = 1

    # --- Indicator Calculations ---
    # Volume SMA for filter
    vol_sma_9 = df['volume'].rolling(9).mean()

    # ATR (Wilder)
    def wilder_atr(high, low, close, length):
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(np.nan, index=tr.index)
        atr.iloc[length - 1] = tr.iloc[:length].mean()
        multiplier = 1.0 / length
        for i in range(length, len(tr)):
            atr.iloc[i] = atr.iloc[i - 1] * (1 - multiplier) + tr.iloc[i] * multiplier
        return atr

    atr_20 = wilder_atr(df['high'], df['low'], df['close'], 20)
    atr_14 = wilder_atr(df['high'], df['low'], df['close'], 14)

    # ATR filter value
    atr_filter_val = atr_20 / 1.5

    # Trend SMA
    sma_54 = df['close'].rolling(54).mean()

    # FVG conditions
    bfvg = (df['low'] > df['high'].shift(2))  # Bullish FVG
    sfvg = (df['high'] < df['low'].shift(2))  # Bearish FVG

    # Filters (assuming enabled as per defaults)
    vol_filt = df['volume'].shift(1) > vol_sma_9 * 1.5
    atr_filt_bull = (df['low'] - df['high'].shift(2) > atr_filter_val)
    atr_filt_bear = (df['low'].shift(2) - df['high'] > atr_filter_val)
    loc_rising = sma_54 > sma_54.shift(1)
    bull_trend_filt = loc_rising
    bear_trend_filt = ~loc_rising

    # Combined FVG with filters
    bull_fvg = bfvg & vol_filt & atr_filt_bull & bull_trend_filt
    bear_fvg = sfvg & vol_filt & atr_filt_bear & bear_trend_filt

    # --- Time Window Check ---
    # Morning: 08:00-09:55 London time
    # Afternoon: 14:00-16:55 London time
    # Friday morning blocked
    london_offsets = {  # Approximate UTC offsets (Pine uses exchange timezone)
        0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1  # Sunday=0..Saturday=6 mapping
    }

    morning_start, morning_end = 8 * 60, 9 * 60 + 55  # minutes from midnight UTC
    afternoon_start, afternoon_end = 14 * 60, 16 * 60 + 55

    in_trading_window = pd.Series(False, index=df.index)
    for i in df.index:
        ts = df['time'].iloc[i]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        # London time: UTC (simplified, ignoring DST fully)
        london_hour = (dt.hour + 0) % 24
        london_min = dt.minute
        minutes = london_hour * 60 + london_min
        dayofweek = dt.weekday()  # Monday=0, Sunday=6

        in_morning = morning_start <= minutes <= morning_end
        in_afternoon = afternoon_start <= minutes <= afternoon_end
        is_friday = (dayofweek == 4)  # Friday

        in_trading_window.iloc[i] = (in_morning or in_afternoon) and not is_friday

    # --- Sharp Turn Detection State ---
    last_fvg = 0  # 1=bullish, -1=bearish, 0=none
    consecutive_bull = 0
    consecutive_bear = 0

    # --- Entry Loop ---
    for i in range(2, len(df)):
        row = df.iloc[i]

        # Skip if in trading window
        if in_trading_window.iloc[i]:
            continue

        # Skip if indicators NaN
        if pd.isna(sma_54.iloc[i]) or pd.isna(atr_14.iloc[i]):
            continue

        bull_sig = bull_fvg.iloc[i]
        bear_sig = bear_fvg.iloc[i]

        # Update consecutive counters
        if bull_sig:
            consecutive_bull += 1
            consecutive_bear = 0
        elif bear_sig:
            consecutive_bear += 1
            consecutive_bull = 0
        else:
            consecutive_bull = 0
            consecutive_bear = 0

        # Entry conditions: Sharp turn (FVG type change)
        long_cond = bull_sig and last_fvg == -1
        short_cond = bear_sig and last_fvg == 1

        if long_cond:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(row['time']),
                'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(row['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['close']),
                'raw_price_b': float(row['close'])
            })
            trade_num += 1
            last_fvg = 1
            consecutive_bull = 0
            consecutive_bear = 0

        elif short_cond:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(row['time']),
                'entry_time': datetime.fromtimestamp(row['time'], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(row['close']),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(row['close']),
                'raw_price_b': float(row['close'])
            })
            trade_num += 1
            last_fvg = -1
            consecutive_bull = 0
            consecutive_bear = 0

        # Update last FVG for non-entry cases
        if bull_sig and not long_cond:
            last_fvg = 1
        elif bear_sig and not short_cond:
            last_fvg = -1

    return entries