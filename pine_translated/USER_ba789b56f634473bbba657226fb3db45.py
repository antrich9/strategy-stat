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

    def wilder_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def wilder_atr(high, low, close, period):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr

    def calc_sma(series, length):
        return series.rolling(length).mean()

    def calc_ema(series, length):
        return series.ewm(span=length, adjust=False).mean()

    # Determine 4H period length in bars (assuming 15min chart = 16 bars per 4H)
    # Try to auto-detect timeframe
    if len(df) >= 2:
        avg_bar_ms = (df['time'].iloc[1] - df['time'].iloc[0])
        if avg_bar_ms > 0:
            bars_per_4h = int(4 * 60 * 60 * 1000 / avg_bar_ms)
        else:
            bars_per_4h = 16
    else:
        bars_per_4h = 16

    # Resample to 4H
    df_4h = df.copy()
    df_4h['period_4h'] = df_4h.index // bars_per_4h
    ohlc_4h = df_4h.groupby('period_4h').agg({
        'time': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # Volume filter: volume[1] > SMA(volume, 9) * 1.5
    vol_sma_9 = calc_sma(ohlc_4h['volume'], 9)
    volfilt = ohlc_4h['volume'].shift(1) > vol_sma_9 * 1.5

    # ATR filter: (low - high[2] > atr_4h) or (low[2] - high > atr_4h)
    atr_4h = wilder_atr(ohlc_4h['high'], ohlc_4h['low'], ohlc_4h['close'], 20) / 1.5
    atrfilt = ((ohlc_4h['low'] - ohlc_4h['high'].shift(2) > atr_4h) |
               (ohlc_4h['low'].shift(2) - ohlc_4h['high'] > atr_4h))

    # Trend filter: SMA(close, 54) > previous SMA
    sma_54 = calc_sma(ohlc_4h['close'], 54)
    loc_bull = sma_54 > sma_54.shift(1)
    loc_bear = ~loc_bull

    # Bullish FVG: low > high[2]
    bfvg1 = ohlc_4h['low'] > ohlc_4h['high'].shift(2)
    # Bearish FVG: high < low[2]
    sfvg1 = ohlc_4h['high'] < ohlc_4h['low'].shift(2)

    # Combined conditions with filters
    bfvg_cond = bfvg1 & volfilt & atrfilt & loc_bull
    sfvg_cond = sfvg1 & volfilt & atrfilt & loc_bear

    # Track last FVG type for sharp turn detection
    last_fvg = 0  # 0 = none, 1 = bullish, -1 = bearish

    entries = []
    trade_num = 1

    # Iterate 4H bars
    for i in range(len(ohlc_4h)):
        if pd.isna(bfvg_cond.iloc[i]) or pd.isna(sfvg_cond.iloc[i]):
            continue

        if bfvg_cond.iloc[i]:
            if last_fvg == -1:
                # Bullish Sharp Turn - LONG entry
                bar_idx_4h = ohlc_4h.index[i]
                entry_ts = int(ohlc_4h['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(ohlc_4h['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(ohlc_4h['close'].iloc[i]),
                    'raw_price_b': float(ohlc_4h['close'].iloc[i])
                })
                trade_num += 1
            last_fvg = 1
        elif sfvg_cond.iloc[i]:
            if last_fvg == 1:
                # Bearish Sharp Turn - SHORT entry
                bar_idx_4h = ohlc_4h.index[i]
                entry_ts = int(ohlc_4h['time'].iloc[i])
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat(),
                    'entry_price_guess': float(ohlc_4h['close'].iloc[i]),
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': float(ohlc_4h['close'].iloc[i]),
                    'raw_price_b': float(ohlc_4h['close'].iloc[i])
                })
                trade_num += 1
            last_fvg = -1

    return entries