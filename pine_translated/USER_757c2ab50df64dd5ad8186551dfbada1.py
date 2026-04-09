import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['period_4h'] = df['datetime'].dt.floor('4h')
    agg_4h = df.groupby('period_4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).reset_index()
    agg_4h['time'] = agg_4h['period_4h'].astype('int64') // 10**9
    agg_4h['high_2'] = agg_4h['high'].shift(2)
    agg_4h['low_2'] = agg_4h['low'].shift(2)
    vol_sma = agg_4h['volume'].ewm(span=9, adjust=False).mean()
    vol_filter = vol_sma * 1.5
    tr1 = agg_4h['high'] - agg_4h['low']
    tr2 = abs(agg_4h['high'] - agg_4h['close'].shift(1))
    tr3 = abs(agg_4h['low'] - agg_4h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=19, adjust=False).mean()
    atr_filter_val = atr / 1.5
    loc1 = agg_4h['close'].ewm(span=54, adjust=False).mean()
    loc21 = loc1 > loc1.shift(1)
    locfiltb = loc21
    locfilts = ~loc21
    bull_fvg = (agg_4h['low'] > agg_4h['high_2']) & (agg_4h['volume'].shift(1) > vol_filter.shift(1)) & (((agg_4h['low'].shift(1) - agg_4h['high'].shift(3)) > atr_filter_val.shift(1)) | ((agg_4h['low'].shift(3) - agg_4h['high'].shift(1)) > atr_filter_val.shift(1))) & locfiltb
    bear_fvg = (agg_4h['high'] < agg_4h['low_2']) & (agg_4h['volume'].shift(1) > vol_filter.shift(1)) & (((agg_4h['low'].shift(1) - agg_4h['high'].shift(3)) > atr_filter_val.shift(1)) | ((agg_4h['low'].shift(3) - agg_4h['high'].shift(1)) > atr_filter_val.shift(1))) & locfilts
    last_fvg = 0
    entries_list = []
    trade_num = 1
    for i in range(len(agg_4h)):
        if i < 3 or pd.isna(agg_4h['high_2'].iloc[i]):
            continue
        current_bull = bull_fvg.iloc[i] if not pd.isna(bull_fvg.iloc[i]) else False
        current_bear = bear_fvg.iloc[i] if not pd.isna(bear_fvg.iloc[i]) else False
        if current_bull and last_fvg == -1:
            ts = int(agg_4h['time'].iloc[i])
            entries_list.append({'trade_num': trade_num, 'direction': 'long', 'entry_ts': ts, 'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(), 'entry_price_guess': float(agg_4h['close'].iloc[i]), 'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0, 'raw_price_a': float(agg_4h['close'].iloc[i]), 'raw_price_b': float(agg_4h['close'].iloc[i])})
            trade_num += 1
            last_fvg = 1
        elif current_bear and last_fvg == 1:
            ts = int(agg_4h['time'].iloc[i])
            entries_list.append({'trade_num': trade_num, 'direction': 'short', 'entry_ts': ts, 'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(), 'entry_price_guess': float(agg_4h['close'].iloc[i]), 'exit_ts': 0, 'exit_time': '', 'exit_price_guess': 0.0, 'raw_price_a': float(agg_4h['close'].iloc[i]), 'raw_price_b': float(agg_4h['close'].iloc[i])})
            trade_num += 1
            last_fvg = -1
        elif current_bull:
            last_fvg = 1
        elif current_bear:
            last_fvg = -1
    return entries_list