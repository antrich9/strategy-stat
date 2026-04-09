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
    # Aggregate 15m data into 4H candles
    four_hours = 4 * 3600
    df = df.copy()
    df['tfour_period'] = df['time'] // four_hours

    agg_4h = df.groupby('tfour_period').agg(
        four_open=('close', 'first'),
        four_high=('high', 'max'),
        four_low=('low', 'min'),
        four_close=('close', 'last'),
        four_vol=('volume', 'sum')
    ).reset_index()

    # Calculate 4H indicators
    # Volume filter: volume[1] > sma(volume, 9) * 1.5
    agg_4h['vol_sma_9'] = agg_4h['four_vol'].rolling(9).mean()
    agg_4h['vol_filt_4h'] = agg_4h['four_vol'].shift(1) > agg_4h['vol_sma_9'] * 1.5

    # ATR filter: (low - high[2] > atr_4h) or (low[2] - high > atr_4h)
    high = agg_4h['four_high']
    low = agg_4h['four_low']
    close = agg_4h['four_close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_len = 20
    atr = np.zeros(len(agg_4h))
    atr[atr_len - 1] = tr.iloc[:atr_len].sum()
    multiplier = 1.0 / atr_len
    for i in range(atr_len, len(agg_4h)):
        atr[i] = atr[i - 1] * (1 - multiplier) + tr.iloc[i] * multiplier
    agg_4h['atr_4h'] = pd.Series(atr, index=agg_4h.index)
    atr_filt_val = agg_4h['atr_4h'] / 1.5

    # Trend filter: SMA(close, 54) > SMA(close, 54)[1]
    agg_4h['sma_54'] = agg_4h['four_close'].rolling(54).mean()
    loc1 = agg_4h['sma_54']
    loc1_prev = loc1.shift(1)
    locfilt_bull = loc1 > loc1_prev
    locfilt_bear = ~(loc1 > loc1_prev)

    # ATR filter condition for 4H
    gap_up = low - high.shift(2)
    gap_down = low.shift(2) - high
    agg_4h['atr_filt_4h'] = (gap_up > atr_filt_val) | (gap_down > atr_filt_val)

    # Bullish FVG: low > high[2] (with filters)
    bull_fvg_4h = (
        (low > high.shift(2)) &
        agg_4h['vol_filt_4h'] &
        agg_4h['atr_filt_4h'] &
        locfilt_bull
    )

    # Bearish FVG: high < low[2] (with filters)
    bear_fvg_4h = (
        (high < low.shift(2)) &
        agg_4h['vol_filt_4h'] &
        agg_4h['atr_filt_4h'] &
        locfilt_bear
    )

    agg_4h['bull_fvg'] = bull_fvg_4h.astype(float)
    agg_4h['bear_fvg'] = bear_fvg_4h.astype(float)

    # Merge 4H indicators to 15m data
    df = df.merge(agg_4h[['tfour_period', 'bull_fvg', 'bear_fvg']], on='tfour_period', how='left')

    # Detect new 4H candles in 15m data
    is_new_4h = df['tfour_period'] != df['tfour_period'].shift()

    entries = []
    trade_num = 1
    last_fvg = 0

    for i in range(len(df)):
        if not is_new_4h.iloc[i]:
            continue

        ts = df['time'].iloc[i]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        time_minutes = hour * 60 + minute

        # London time window 10:45 to 11:45
        if not (645 <= time_minutes <= 705):
            continue

        prev_last_fvg = last_fvg

        # Update last_fvg based on previous 4H candle
        if i > 0:
            j = i - 1
            bull_fvg = agg_4h['bull_fvg'].iloc[j] > 0 if j < len(agg_4h) else False
            bear_fvg = agg_4h['bear_fvg'].iloc[j] > 0 if j < len(agg_4h) else False
            if bull_fvg:
                last_fvg = 1
            elif bear_fvg:
                last_fvg = -1

        # Sharp turn detection
        bull_fvg_cur = df['bull_fvg'].iloc[i] > 0
        bear_fvg_cur = df['bear_fvg'].iloc[i] > 0

        if prev_last_fvg == -1 and bull_fvg_cur:
            # Bullish Sharp Turn - Long
            entry_ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i], 'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
        elif prev_last_fvg == 1 and bear_fvg_cur:
            # Bearish Sharp Turn - Short
            entry_ts = df['time'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i], 'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1

        if bull_fvg_cur:
            last_fvg = 1
        elif bear_fvg_cur:
            last_fvg = -1

    return entries