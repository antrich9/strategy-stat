import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['datetime'].dt.date

    daily_ohlc = df.groupby('date').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last')
    )

    daily_ohlc['prev_high'] = daily_ohlc['high'].shift(1)
    daily_ohlc['prev_low'] = daily_ohlc['low'].shift(1)
    daily_ohlc['high_2d'] = daily_ohlc['high'].shift(2)
    daily_ohlc['low_2d'] = daily_ohlc['low'].shift(2)
    daily_ohlc['atr'] = (daily_ohlc['high'] - daily_ohlc['low']).rolling(20).mean() / 1.5

    df = df.merge(daily_ohlc[['prev_high', 'prev_low', 'high_2d', 'low_2d', 'atr']], left_on='date', right_index=True, how='left')

    df['vol_sma'] = df['volume'].rolling(9).mean()
    df['volfilt'] = df['volume'].shift(1) > df['vol_sma'] * 1.5

    df['atrfilt'] = ((df['low'] - df['high_2d'].shift(1) > df['atr'].shift(1)) | 
                     (df['low_2d'].shift(1) - df['high'] > df['atr'].shift(1)))

    df['loc'] = df['close'].rolling(54).mean()
    df['loc_prev'] = df['loc'].shift(1)
    df['locfiltb'] = df['loc'] > df['loc_prev']
    df['locfilts'] = df['loc'] < df['loc_prev']

    df['bfvg'] = (df['low'] > df['high_2d'].shift(1)) & df['volfilt'] & df['atrfilt'] & df['locfiltb']
    df['sfvg'] = (df['high'] < df['low_2d'].shift(1)) & df['volfilt'] & df['atrfilt'] & df['locfilts']

    daily_ohlc['is_swing_high'] = ((daily_ohlc['prev_high'] < daily_ohlc['high_2d']) & 
                                   (daily_ohlc['high'].shift(3) < daily_ohlc['high_2d']) & 
                                   (daily_ohlc['high'].shift(4) < daily_ohlc['high_2d']))
    daily_ohlc['is_swing_low'] = ((daily_ohlc['prev_low'] > daily_ohlc['low_2d']) & 
                                  (daily_ohlc['low'].shift(3) > daily_ohlc['low_2d']) & 
                                  (daily_ohlc['low'].shift(4) > daily_ohlc['low_2d']))

    df = df.merge(daily_ohlc[['is_swing_high', 'is_swing_low']], left_on='date', right_index=True, how='left')

    isBullishLeg = False
    isBearishLeg = False
    bullfvghigh = np.nan
    bearfvglow = np.nan
    bulltap = 0
    beartap = 0
    lastSwingType = 'none'
    bullishFVG = False
    bearishFVG = False

    entries = []
    trade_num = 1

    for i in range(len(df)):
        row = df.iloc[i]

        if row['is_swing_high']:
            lastSwingType = 'dailyHigh'
        if row['is_swing_low']:
            lastSwingType = 'dailyLow'

        if row['bfvg'] and lastSwingType == 'dailyLow' and not pd.isna(row['low']):
            bullishFVG = True
            bearishFVG = False
            isBullishLeg = True
            isBearishLeg = False
            bullfvghigh = row['low']
            bulltap = 0

        if row['sfvg'] and lastSwingType == 'dailyHigh' and not pd.isna(row['high']):
            bearishFVG = True
            bullishFVG = False
            isBearishLeg = True
            isBullishLeg = False
            bearfvglow = row['high']
            beartap = 0

        if i > 0:
            prev_row = df.iloc[i-1]

            if not pd.isna(bullfvghigh) and prev_row['low'] >= bullfvghigh and row['low'] < bullfvghigh:
                if isBearishLeg:
                    bulltap = 1000
                if bulltap < 1000:
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'long',
                        'entry_ts': int(row['time']),
                        'entry_time': row['datetime'].replace(tzinfo=timezone.utc).isoformat(),
                        'entry_price_guess': float(row['close']),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(row['close']),
                        'raw_price_b': float(row['close'])
                    })
                    trade_num += 1
                bulltap += 1

            if not pd.isna(bearfvglow) and prev_row['high'] <= bearfvglow and row['high'] > bearfvglow:
                if isBullishLeg:
                    beartap = 1000
                if beartap < 1000:
                    entries.append({
                        'trade_num': trade_num,
                        'direction': 'short',
                        'entry_ts': int(row['time']),
                        'entry_time': row['datetime'].replace(tzinfo=timezone.utc).isoformat(),
                        'entry_price_guess': float(row['close']),
                        'exit_ts': 0,
                        'exit_time': '',
                        'exit_price_guess': 0.0,
                        'raw_price_a': float(row['close']),
                        'raw_price_b': float(row['close'])
                    })
                    trade_num += 1
                beartap += 1

        if isBearishLeg and not bullishFVG:
            bulltap = 1000
        if isBullishLeg and not bearishFVG:
            beartap = 1000

    return entries