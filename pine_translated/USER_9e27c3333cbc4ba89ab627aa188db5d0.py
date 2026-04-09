def generate_entries(df: pd.DataFrame) -> list:
    import pandas as pd
    import numpy as np
    from datetime import datetime, timezone

    sensitivity = 150
    fastLength = 20
    slowLength = 40
    channelLength = 20
    mult = 2.0

    close = df['close']

    fast_ema = close.ewm(span=fastLength, adjust=False).mean()
    slow_ema = close.ewm(span=slowLength, adjust=False).mean()
    macd = fast_ema - slow_ema
    macd_prev = macd.shift(1)

    t1 = (macd - macd_prev) * sensitivity

    basis = close.rolling(window=channelLength).mean()
    stdev = close.rolling(window=channelLength).std(ddof=0)
    e1 = (basis + mult * stdev) - (basis - mult * stdev)

    trendUp = t1.clip(lower=0)

    entry_allowed = pd.Series(True, index=df.index, dtype=bool)
    for i in range(1, len(df)):
        if trendUp.iloc[i] < e1.iloc[i]:
            entry_allowed.iloc[i] = True
        else:
            entry_allowed.iloc[i] = entry_allowed.iloc[i-1]

    crossover = (trendUp > e1) & (trendUp.shift(1) <= e1.shift(1))
    entry_signal = entry_allowed & crossover

    entries = []
    trade_num = 1
    for i in range(len(df)):
        if entry_signal.iloc[i]:
            if pd.isna(trendUp.iloc[i]) or pd.isna(e1.iloc[i]) or pd.isna(entry_allowed.iloc[i]):
                continue
            ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1

    return entries