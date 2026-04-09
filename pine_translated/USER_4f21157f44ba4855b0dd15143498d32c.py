import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    time = df['time']
    open_ = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']

    # ATR using Wilder's method
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(20).mean()
    atr = atr.iloc[19]
    for i in range(20, len(atr)):
        if pd.notna(atr.iloc[i]):
            atr.iloc[i] = (atr.iloc[i-1] * 19 + tr.iloc[i]) / 20

    # Order Block detection (obUp/obDown calculated at script start)
    isDown = close < open_
    isUp = close > open_
    obUp = isDown.shift(1) & isUp & (close.shift(1) > high.shift(2))
    obDown = isUp.shift(1) & isDown & (close.shift(1) < low.shift(2))

    # Fair Value Gap detection
    fvgUp = low > high.shift(2)
    fvgDown = high < low.shift(2)

    # Location/trend filter
    loc_sma = close.rolling(54).mean()
    locfiltb = loc_sma > loc_sma.shift(1)
    locfilts = loc_sma < loc_sma.shift(1)

    # Top Imbalance (Breakaway FVG)
    top_imb_bway = (low.shift(2) <= open_.shift(1)) & (high >= close.shift(1)) & (close < low.shift(1))
    top_imb_xbway = (low.shift(2) <= open_.shift(1)) & (high >= close.shift(1)) & (close > low.shift(1))

    # Filters
    volfilt = (volume.shift(1) > volume.rolling(9).mean() * 1.5) | volume.shift(1).isna()
    atrfilt = ((low - high.shift(2)) > atr) | (low.shift(2) - high > atr) | atr.isna()

    # Combined FVG conditions
    bfvg = fvgUp & volfilt & atrfilt & locfiltb
    sfvg = fvgDown & volfilt & atrfilt & locfilts

    # Trading windows in London time
    dt_utc = pd.to_datetime(time, unit='s', utc=True)
    london_times = dt_utc.dt.tz_convert('Europe/London')
    hours = london_times.dt.hour
    minutes = london_times.dt.minute
    in_morning = (hours == 7) & (minutes >= 45) | (hours == 8) | (hours == 9) & (minutes <= 44)
    in_afternoon = (hours == 14) & (minutes >= 45) | (hours == 15) | (hours == 16) & (minutes <= 44)
    in_trading_window = in_morning | in_afternoon

    # Long/short conditions
    is_long = obUp & bfvg & locfiltb & top_imb_bway
    is_short = obDown & sfvg & locfilts & top_imb_xbway

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(is_long.iloc[i]) or pd.isna(bfvg.iloc[i]) or pd.isna(locfiltb.iloc[i]) or pd.isna(top_imb_bway.iloc[i]):
            continue
        if pd.isna(is_short.iloc[i]) or pd.isna(sfvg.iloc[i]) or pd.isna(locfilts.iloc[i]) or pd.isna(top_imb_xbway.iloc[i]):
            continue
        if is_long.iloc[i] and in_trading_window.iloc[i]:
            ts = int(time.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1
        elif is_short.iloc[i] and in_trading_window.iloc[i]:
            ts = int(time.iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(close.iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(close.iloc[i]),
                'raw_price_b': float(close.iloc[i])
            })
            trade_num += 1

    return entries