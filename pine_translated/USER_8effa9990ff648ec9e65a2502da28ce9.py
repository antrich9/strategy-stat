import pandas as pd
import numpy as np
from datetime import datetime, timezone

def calculate_wilder_rsi(prices: pd.Series, length: int) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/length, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/length, adjust=False).mean()
    return atr

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['ts'] = df['time']

    df_240 = df.set_index('ts').resample('240T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    daily_df = pd.DataFrame(index=df.index)
    daily_df['dailyOpen'] = df_240['open'].reindex(df.index, method='ffill')
    daily_df['dailyHigh'] = df_240['high'].reindex(df.index, method='ffill')
    daily_df['dailyLow'] = df_240['low'].reindex(df.index, method='ffill')
    daily_df['dailyClose'] = df_240['close'].reindex(df.index, method='ffill')

    daily_df['prevDayHigh'] = daily_df['dailyHigh'].shift(1)
    daily_df['prevDayLow'] = daily_df['dailyLow'].shift(1)
    daily_df['dailyHigh1'] = daily_df['dailyHigh'].shift(1)
    daily_df['dailyLow1'] = daily_df['dailyLow'].shift(1)
    daily_df['dailyHigh2'] = daily_df['dailyHigh'].shift(2)
    daily_df['dailyLow2'] = daily_df['dailyLow'].shift(2)

    df['volfilt'] = (df['volume'].shift(1) > df['volume'].shift(1).rolling(9).mean() * 1.5)

    atr = calculate_wilder_atr(df['high'], df['low'], df['close'], 20)
    df['atr2'] = atr / 1.5
    df['atrfilt'] = ((daily_df['dailyLow'] - daily_df['dailyHigh2'] > df['atr2']) |
                     (daily_df['dailyLow2'] - daily_df['dailyHigh'] > df['atr2']))

    loc = df['close'].rolling(54).mean()
    df['locfiltb'] = (loc > loc.shift(1))
    df['locfilts'] = (loc < loc.shift(1))

    bfvg = (daily_df['dailyLow'] > daily_df['dailyHigh2']) & df['volfilt'] & df['atrfilt'] & df['locfiltb']
    sfvg = (daily_df['dailyHigh'] < daily_df['dailyLow2']) & df['volfilt'] & df['atrfilt'] & df['locfilts']

    entries = []
    trade_num = 1

    for i in range(1, len(df)):
        if pd.isna(daily_df['dailyHigh2'].iloc[i]) or pd.isna(daily_df['dailyLow2'].iloc[i]):
            continue
        if bfvg.iloc[i] and not bfvg.iloc[i-1]:
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['ts'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['ts'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        if sfvg.iloc[i] and not sfvg.iloc[i-1]:
            entry_price = df['close'].iloc[i]
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['ts'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['ts'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries