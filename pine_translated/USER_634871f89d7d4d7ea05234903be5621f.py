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
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values
    timestamps = df['time'].values

    n = len(df)
    entries = []
    trade_num = 1

    WARMUP = 100
    if n < WARMUP + 10:
        return entries

    close_series = pd.Series(close)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    volume_series = pd.Series(volume)

    vol_sma_9 = volume_series.rolling(9).mean()
    vol_filt = volume_series.shift(1) > vol_sma_9 * 1.5

    tr1 = high_series - low_series
    tr2 = (high_series - close_series.shift(1)).abs()
    tr3 = (low_series - close_series.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/20, adjust=False).mean()
    atr_val = atr / 1.5
    atr_filt = (low_series - high_series.shift(2) > atr_val) | (low_series.shift(2) - high_series > atr_val)

    trend_sma = close_series.rolling(54).mean()
    locfiltb = trend_sma > trend_sma.shift(1)
    locfilts = trend_sma < trend_sma.shift(1)

    bfvg = low_series > high_series.shift(2)
    sfvg = high_series < low_series.shift(2)

    last_swing_high = np.nan
    last_swing_low = np.nan
    lastSwingType = "none"

    bull_fvg_top = np.nan
    bull_fvg_bottom = np.nan
    bull_fvg_stop = np.nan
    bull_fvg_active = False
    bull_fvg_traded = False
    bull_fvg_bar = 0

    bear_fvg_top = np.nan
    bear_fvg_bottom = np.nan
    bear_fvg_stop = np.nan
    bear_fvg_active = False
    bear_fvg_traded = False
    bear_fvg_bar = 0

    bar_index = 0
    for i in range(WARMUP, n):
        bar_index = i - WARMUP + 1

        h2 = high[i - 2]
        h1 = high[i - 1]
        h3 = high[i - 3] if i >= 3 else high[i - 2]
        h4 = high[i - 4] if i >= 4 else high[i - 2]
        l2 = low[i - 2]
        l1 = low[i - 1]
        l3 = low[i - 3] if i >= 3 else low[i - 2]
        l4 = low[i - 4] if i >= 4 else low[i - 2]

        is_swing_high = h1 < h2 and h3 < h2 and h4 < h2
        is_swing_low = l1 > l2 and l3 > l2 and l4 > l2

        if is_swing_high:
            last_swing_high = h2
            lastSwingType = "high"
        if is_swing_low:
            last_swing_low = l2
            lastSwingType = "low"

        current_bfvg = bfvg.iloc[i]
        current_sfvg = sfvg.iloc[i]
        current_volfilt = vol_filt.iloc[i]
        current_atrfilt = atr_filt.iloc[i]
        current_locfiltb = locfiltb.iloc[i]
        current_locfilts = locfilts.iloc[i]

        isBullishLeg = current_bfvg and lastSwingType == "low"
        isBearishLeg = current_sfvg and lastSwingType == "high"

        if current_bfvg and isBullishLeg and current_volfilt and current_atrfilt and current_locfiltb:
            bull_fvg_top = low[i]
            bull_fvg_bottom = high[i - 2]
            bull_fvg_stop = low[i - 2]
            bull_fvg_active = True
            bull_fvg_traded = False
            bull_fvg_bar = bar_index

        if current_sfvg and isBearishLeg and current_volfilt and current_atrfilt and current_locfilts:
            bear_fvg_top = low[i - 2]
            bear_fvg_bottom = high[i]
            bear_fvg_stop = high[i - 2]
            bear_fvg_active = True
            bear_fvg_traded = False
            bear_fvg_bar = bar_index

        if not bull_fvg_active and not bear_fvg_active:
            if current_bfvg and isBullishLeg and current_volfilt and current_atrfilt and current_locfiltb:
                bull_fvg_top = low[i]
                bull_fvg_bottom = high[i - 2]
                bull_fvg_stop = low[i - 2]
                bull_fvg_active = True
                bull_fvg_traded = False
                bull_fvg_bar = bar_index

            if current_sfvg and isBearishLeg and current_volfilt and current_atrfilt and current_locfilts:
                bear_fvg_top = low[i - 2]
                bear_fvg_bottom = high[i]
                bear_fvg_stop = high[i - 2]
                bear_fvg_active = True
                bear_fvg_traded = False
                bear_fvg_bar = bar_index

        if bull_fvg_active and not bull_fvg_traded and bar_index > bull_fvg_bar:
            retest = low[i] <= bull_fvg_top and high[i] >= bull_fvg_bottom
            if retest:
                entry_ts = timestamps[i]
                entry_price_guess = bull_fvg_top
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price_guess,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price_guess,
                    'raw_price_b': entry_price_guess
                })
                trade_num += 1
                bull_fvg_traded = True
                bull_fvg_active = False

        if bear_fvg_active and not bear_fvg_traded and bar_index > bear_fvg_bar:
            retest = high[i] >= bear_fvg_bottom and low[i] <= bear_fvg_top
            if retest:
                entry_ts = timestamps[i]
                entry_price_guess = bear_fvg_bottom
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                    'entry_price_guess': entry_price_guess,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price_guess,
                    'raw_price_b': entry_price_guess
                })
                trade_num += 1
                bear_fvg_traded = True
                bear_fvg_active = False

    return entries