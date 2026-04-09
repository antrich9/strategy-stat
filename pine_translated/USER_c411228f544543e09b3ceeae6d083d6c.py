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
    if len(df) < 20:
        return []

    ts = df['time'].iloc[0]
    is_ms = ts > 1e12
    if is_ms:
        df_temp = df.copy()
        df_temp['time'] = df_temp['time'] // 1000
    else:
        df_temp = df.copy()

    # Determine if 4H or need to aggregate
    time_diffs = df_temp['time'].diff().dropna()
    if len(time_diffs) == 0:
        return []
    avg_diff = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()

    # Aggregate to 4H if input is lower timeframe (avg_diff < 14400)
    if avg_diff < 14400:
        df_temp['time_ts'] = pd.to_datetime(df_temp['time'], unit='s', utc=True)
        df_temp['4h_period'] = df_temp['time_ts'].dt.to_period('4H')
        agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        df_4h = df_temp.groupby('4h_period').agg(agg_dict).reset_index(drop=True)
        # Keep last timestamp of each 4H period for time column
        df_4h_time = df_temp.groupby('4h_period')['time'].max()
        df_4h['time'] = df_4h_time.values
    else:
        df_4h = df_temp.reset_index(drop=True)

    if len(df_4h) < 10:
        return []

    close_4h = df_4h['close']
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    volume_4h = df_4h['volume']

    # Volume filter: volume_4h[1] > sma(volume_4h, 9) * 1.5
    vol_sma_9 = volume_4h.rolling(9).mean()
    vol_filt = volume_4h.shift(1) > vol_sma_9 * 1.5

    # ATR filter using Wilder ATR (14-period standard)
    atr_len = 14
    tr = pd.concat([high_4h - low_4h, (high_4h - close_4h.shift(1)).abs(), (low_4h - close_4h.shift(1)).abs()], axis=1).max(axis=1)
    atr_4h = tr.ewm(alpha=1/atr_len, adjust=False).mean() / 1.5

    # Check on bar i: low[i] - high[i-2] > atr[i] OR low[i-2] - high[i] > atr[i]
    # Shift for previous values
    high_lag2 = high_4h.shift(2)
    low_lag2 = low_4h.shift(2)
    atr_lag0 = atr_4h  # atr on current bar

    # Detect FVG at bar i: need low[i], high[i-2], atr[i]
    # We need to ensure we have valid values at i and i-2
    # atrfilt1[i] = (low_4h[i] - high_4h[i-2] > atr_4h[i]) OR (low_4h[i-2] - high_4h[i] > atr_4h[i])
    fvg_gap_up = low_4h - high_lag2
    fvg_gap_down = low_lag2 - high_4h
    atr_filt = (fvg_gap_up > atr_lag0) | (fvg_gap_down > atr_lag0)

    # Trend filter using 4H SMA(54)
    trend_sma = close_4h.rolling(54).mean()
    trend_up = trend_sma > trend_sma.shift(1)
    trend_dn = ~trend_up

    # Bullish FVG: low > high[2] AND vol_filt AND atr_filt AND trend_up
    # Bearish FVG: high < low[2] AND vol_filt AND atr_filt AND trend_dn
    bull_fvg = (low_4h > high_lag2) & vol_filt & atr_filt & trend_up
    bear_fvg = (high_4h < low_lag2) & vol_filt & atr_filt & trend_dn

    # Build boolean series aligned with df_4h index
    bull_fvg = bull_fvg.reindex(df_4h.index).fillna(False)
    bear_fvg = bear_fvg.reindex(df_4h.index).fillna(False)

    entries = []
    trade_num = 1
    last_fvg = 0  # 0=none, 1=bullish, -1=bearish

    for i in range(2, len(df_4h)):
        if pd.isna(bull_fvg.iloc[i]) or pd.isna(bear_fvg.iloc[i]):
            continue

        curr_bull = bull_fvg.iloc[i]
        curr_bear = bear_fvg.iloc[i]

        if curr_bull and last_fvg == -1:
            ts_val = int(df_4h['time'].iloc[i])
            if is_ms:
                ts_val = ts_val * 1000
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts_val,
                'entry_time': datetime.fromtimestamp(df_4h['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df_4h['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df_4h['close'].iloc[i]),
                'raw_price_b': float(df_4h['close'].iloc[i])
            })
            trade_num += 1
            last_fvg = 1

        if curr_bear and last_fvg == 1:
            ts_val = int(df_4h['time'].iloc[i])
            if is_ms:
                ts_val = ts_val * 1000
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts_val,
                'entry_time': datetime.fromtimestamp(df_4h['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df_4h['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df_4h['close'].iloc[i]),
                'raw_price_b': float(df_4h['close'].iloc[i])
            })
            trade_num += 1
            last_fvg = -1

        # Update last_fvg even if not sharp turn
        if curr_bull and last_fvg != 1:
            last_fvg = 1
        elif curr_bear and last_fvg != -1:
            last_fvg = -1

    return entries