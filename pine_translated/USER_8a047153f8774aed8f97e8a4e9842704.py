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
    # Volume filter: volume[1] > ta.sma(volume, 9) * 1.5
    vol_sma = df['volume'].rolling(9).mean()
    volfilt = df['volume'].shift(1) > vol_sma * 1.5

    # ATR filter: (low - high[2] > atr) or (low[2] - high > atr)
    def wilder_atr(high_arr, low_arr, close_arr, length=20):
        tr1 = high_arr - low_arr
        tr2 = (high_arr - close_arr.shift(1)).abs()
        tr3 = (low_arr - close_arr.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/length, adjust=False).mean()
        return atr

    atr = wilder_atr(df['high'], df['low'], df['close'], 20) / 1.5
    atrfilt_long = (df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr)
    atrfilt_short = (df['low'] - df['high'].shift(2) > atr) | (df['low'].shift(2) - df['high'] > atr)
    atrfilt = atrfilt_long | atrfilt_short

    # Trend filter: sma(close, 54), loc2 = loc > loc[1]
    loc = df['close'].rolling(54).mean()
    loc2 = loc > loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2

    # Bullish FVG: low > high[2] and volfilt and atrfilt and locfiltb
    bfvg = (df['low'] > df['high'].shift(2)) & volfilt & atrfilt & locfiltb

    # Bearish FVG: high < low[2] and volfilt and atrfilt and locfilts
    sfvg = (df['high'] < df['low'].shift(2)) & volfilt & atrfilt & locfilts

    # Order Block conditions
    isUp = df['close'] > df['open']
    isDown = df['close'] < df['open']
    obUp = isDown.shift(1) & isUp & (df['close'] > df['high'].shift(1))
    obDown = isUp.shift(1) & isDown & (df['close'] < df['low'].shift(1))
    fvgUp = df['low'] > df['high'].shift(2)
    fvgDown = df['high'] < df['low'].shift(2)

    # Stacked OB+FVG conditions
    stacked_bullish = obUp & fvgUp
    stacked_bearish = obDown & fvgDown

    # Entry conditions: FVG with confirmation (price reacts from the zone)
    # Bullish entry: FVG forms, then price pulls back and bounces
    bullish_entry = stacked_bullish | bfvg
    bearish_entry = stacked_bearish | sfvg

    # Alternative entries based on imbalances (for variety)
    top_imb = (df['low'].shift(2) <= df['open'].shift(1)) & (df['high'] >= df['close'].shift(1))
    bot_imb = (df['high'].shift(2) >= df['open'].shift(1)) & (df['low'] <= df['close'].shift(1))

    # Combine all bullish signals
    long_signal = bullish_entry | (top_imb & locfiltb)
    short_signal = bearish_entry | (bot_imb & locfilts)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if i < 2:
            continue
        if pd.isna(df['close'].iloc[i]):
            continue
        if long_signal.iloc[i] if hasattr(long_signal.iloc[i], '__bool__') else long_signal[i]:
            entry_price = df['close'].iloc[i]
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
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
        elif short_signal.iloc[i] if hasattr(short_signal.iloc[i], '__bool__') else short_signal[i]:
            entry_price = df['close'].iloc[i]
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
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

    return entries