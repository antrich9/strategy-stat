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
    results = []
    trade_num = 1

    # Extract time components
    df = df.copy()
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    df['dayofweek'] = df['dt'].dt.dayofweek  # 0=Monday, 6=Sunday

    # London time windows: 8:00-9:55 and 14:00-16:55
    morning_start = (df['hour'] == 8) & (df['minute'] >= 0)
    morning_end = (df['hour'] == 9) & (df['minute'] <= 55)
    isWithinMorningWindow = morning_start | morning_end

    afternoon_start = (df['hour'] == 14) & (df['minute'] >= 0)
    afternoon_end = (df['hour'] == 16) & (df['minute'] <= 55)
    isWithinAfternoonWindow = afternoon_start | afternoon_end

    isWithinTimeWindow = isWithinMorningWindow | isWithinAfternoonWindow
    isFridayMorningWindow = (df['dayofweek'] == 4) & isWithinMorningWindow

    # EMA calculations
    shortEmaLength = 50
    midEmaLength = 100
    longEmaLength = 200

    df['shortEma'] = df['close'].ewm(span=shortEmaLength, adjust=False).mean()
    df['midEma'] = df['close'].ewm(span=midEmaLength, adjust=False).mean()
    df['longEma'] = df['close'].ewm(span=longEmaLength, adjust=False).mean()

    # MTF calculations (daily timeframe)
    daily_df = df.resample('1D', on='dt').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

    daily_df['shortEma'] = daily_df['close'].ewm(span=shortEmaLength, adjust=False).mean()
    daily_df['longEma'] = daily_df['close'].ewm(span=longEmaLength, adjust=False).mean()
    daily_df['pivotSource'] = (daily_df['high'] + daily_df['low'] + daily_df['close']) / 3

    # Reindex daily data to 15m data
    df['date'] = df['dt'].dt.date
    daily_df = daily_df.reset_index().set_index('dt')
    df = df.set_index('dt')

    mtfShortEma = df['date'].map(daily_df['shortEma'])
    mtfLongEma = df['date'].map(daily_df['longEma'])
    mtfPivotSource = df['date'].map(daily_df['pivotSource'])
    mtfHigh = df['date'].map(daily_df['high'])
    mtfLow = df['date'].map(daily_df['low'])

    df['mtfShortEma'] = mtfShortEma
    df['mtfLongEma'] = mtfLongEma
    df['mtfPivotSource'] = mtfPivotSource
    df['mtfHigh'] = mtfHigh
    df['mtfLow'] = mtfLow

    df = df.reset_index()

    # Pivot calculations
    df['pivotPP'] = df['mtfPivotSource']
    df['pivotS1'] = df['pivotPP'] - (df['mtfHigh'] - df['mtfLow']) / 2
    df['pivotR1'] = df['pivotPP'] + (df['mtfHigh'] - df['mtfLow']) / 2
    df['pivotS2'] = df['pivotPP'] - (df['mtfHigh'] - df['mtfLow'])
    df['pivotR2'] = df['pivotPP'] + (df['mtfHigh'] - df['mtfLow'])

    # Wilder RSI implementation
    def wilder_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # Wilder ATR implementation
    def wilder_atr(high, low, close, period):
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr

    df['atr1'] = wilder_atr(df['high'], df['low'], df['close'], 14)
    df['atr'] = wilder_atr(df['high'], df['low'], df['close'], 20)

    # Rejection conditions
    bullishRejection = (
        (df['low'] < df['pivotS1'].shift(1)) & (df['low'].shift(1) >= df['pivotS1']) |
        (df['low'] < df['pivotS2'].shift(1)) & (df['low'].shift(1) >= df['pivotS2'])
    )

    bearishRejection = (
        (df['high'] > df['pivotR1'].shift(1)) & (df['high'].shift(1) <= df['pivotR1']) |
        (df['high'] > df['pivotR2'].shift(1)) & (df['high'].shift(1) <= df['pivotR2'])
    )

    # Candle conditions
    bullishCandle = df['close'] > df['open']
    bearishCandle = df['close'] < df['open']

    # validbuy and validsell conditions
    validbuy_cond = (df['shortEma'] > df['longEma']) & (df['mtfShortEma'] > df['mtfLongEma']) & bullishRejection
    validsell_cond = (df['shortEma'] < df['longEma']) & (df['mtfShortEma'] < df['mtfLongEma']) & bearishRejection

    df['validbuy'] = validbuy_cond
    df['validsell'] = validsell_cond
    df['bullishRejection'] = bullishRejection
    df['bearishRejection'] = bearishRejection
    df['bullishCandle'] = bullishCandle
    df['bearishCandle'] = bearishCandle
    df['isWithinTimeWindow'] = isWithinTimeWindow.values
    df['isFridayMorningWindow'] = isFridayMorningWindow.values

    # CanTrade: only one trade per day
    df['date_str'] = df['dt'].dt.date.astype(str)
    lastTradeDay = None
    df['canTrade'] = False
    for i in range(len(df)):
        current_date = df['date_str'].iloc[i]
        if lastTradeDay is None or current_date > lastTradeDay:
            df.iloc[i, df.columns.get_loc('canTrade')] = True
            lastTradeDay = current_date

    # Entry conditions
    long_entry_cond = df['validbuy'] & df['canTrade'] & df['bearishCandle']
    short_entry_cond = df['validsell'] & df['canTrade'] & df['bullishCandle']

    # No position check (simulated)
    position_open = False

    for i in range(len(df)):
        if pd.isna(df['shortEma'].iloc[i]) or pd.isna(df['longEma'].iloc[i]):
            continue
        if pd.isna(df['mtfShortEma'].iloc[i]) or pd.isna(df['mtfLongEma'].iloc[i]):
            continue

        # Long entry
        if not position_open and df['isWithinTimeWindow'].iloc[i] and not df['isFridayMorningWindow'].iloc[i]:
            if long_entry_cond.iloc[i]:
                entry_price = df['close'].iloc[i]
                entry_ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()

                results.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
                position_open = True

        # Short entry
        if not position_open and df['isWithinTimeWindow'].iloc[i] and not df['isFridayMorningWindow'].iloc[i]:
            if short_entry_cond.iloc[i]:
                entry_price = df['close'].iloc[i]
                entry_ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()

                results.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
                position_open = True

    return results