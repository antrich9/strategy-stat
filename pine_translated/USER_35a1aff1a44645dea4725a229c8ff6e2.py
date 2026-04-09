import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.sort_values('time').reset_index(drop=True)

    entry_pips = 3.0
    atr_multiplier = 1.5
    atr_length = 14
    adx_threshold = 20
    min_volume_multiplier = 1.2
    require_candle_close = True
    mintick = 0.0001
    entry_pip_value = entry_pips * 10 * mintick

    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/atr_length, adjust=False).mean()

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        up_move = high.iloc[i] - high.iloc[i-1]
        down_move = low.iloc[i-1] - low.iloc[i]
        if up_move > down_move and up_move > 0:
            plus_dm.iloc[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm.iloc[i] = down_move

    plus_dm_smooth = plus_dm.ewm(alpha=1.0/14, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1.0/14, adjust=False).mean()
    di_plus = 100 * (plus_dm_smooth / atr)
    di_minus = 100 * (minus_dm_smooth / atr)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.ewm(alpha=1.0/14, adjust=False).mean()

    vol_sma = df['volume'].rolling(20).mean()
    is_high_volume = df['volume'] > (vol_sma * min_volume_multiplier)

    df['date'] = df['time'].dt.date
    daily_high_expanded = df.groupby('date')['high'].cummax()
    daily_low_expanded = df.groupby('date')['low'].cummin()
    new_day = df['date'].diff().fillna(pd.Timedelta(days=1)) != pd.Timedelta(days=0)
    new_day.iloc[0] = True

    entries = []
    trade_num = 1
    in_trade = False
    current_daily_high = None
    current_daily_low = None
    current_adx = None
    current_vol_sma = None

    for i in range(1, len(df)):
        if new_day.iloc[i]:
            current_daily_high = daily_high_expanded.iloc[i]
            current_daily_low = daily_low_expanded.iloc[i]
            current_adx = adx.iloc[i]
            current_vol_sma = vol_sma.iloc[i]
            in_trade = False

        if new_day.iloc[i] and not in_trade and not pd.isna(atr.iloc[i]) and not pd.isna(adx.iloc[i]):
            buy_point = current_daily_high + entry_pip_value
            sell_point = current_daily_low - entry_pip_value
            is_hv = is_high_volume.iloc[i]
            bullish_breakout = (close.iloc[i] > buy_point and close.iloc[i] > current_daily_high and current_adx > adx_threshold and is_hv)
            bearish_breakout = (close.iloc[i] < sell_point and close.iloc[i] < current_daily_low and current_adx > adx_threshold and is_hv)

            if require_candle_close:
                if bullish_breakout and i > 0:
                    bullish_breakout = close.iloc[i] > buy_point
                if bearish_breakout and i > 0:
                    bearish_breakout = close.iloc[i] < sell_point

            if bullish_breakout:
                ts = int(df['time'].iloc[i].timestamp())
                entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': ts,
                    'entry_time': entry_time_str,
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
                trade_num += 1
                in_trade = True

            if bearish_breakout:
                ts = int(df['time'].iloc[i].timestamp())
                entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': ts,
                    'entry_time': entry_time_str,
                    'entry_price_guess': close.iloc[i],
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close.iloc[i],
                    'raw_price_b': close.iloc[i]
                })
                trade_num += 1
                in_trade = True

    return entries