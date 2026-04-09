import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    # Default parameters (as per the script)
    waveLength = 50
    swingStrength = 5
    useTrendFilter = True
    volMAPeriod = 20
    volMultiplier = 0.0  # 0 means auto
    requireVolSurge = True
    atrPeriod = 14
    atrStopInput = 0.0  # auto
    Rvalue = 1.0
    tpRMultiple = 20.0
    useAllHours = False
    useLondon = True
    useNYAM = True
    useNYPM = False
    riskPercent = 1.0
    maxTradesPerDay = 3

    # Auto set multipliers based on instrument? We don't know instrument, so assume Gold? Or generic.
    # We'll set volMultiplier to 1.3 and atrStopMulti to 2.5 as defaults for Gold.
    # But the script uses auto detection. Since we don't have instrument info, we'll assume Gold.
    # However, the problem doesn't specify. We can set to the defaults for Gold.
    volMultiplier = 1.3 if volMultiplier == 0.0 else volMultiplier
    atrStopMulti = 2.5 if atrStopInput == 0.0 else atrStopInput

    # Compute indicators
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # EMA
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    # Uptrend and downtrend
    uptrend = (ema50 > ema200) & (close > ema50)
    downtrend = (ema50 < ema200) & (close < ema50)
    trendValid = (not useTrendFilter) | uptrend | downtrend

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macdLine = ema12 - ema26
    macdSignal = macdLine.ewm(span=9, adjust=False).mean()
    macdBullish = macdLine > macdSignal
    macdBearish = macdLine < macdSignal

    # Volume
    volMA = volume.rolling(volMAPeriod).mean()
    volumeSurge = volume > (volMA * volMultiplier)
    volConfirmed = (not requireVolSurge) | volumeSurge

    # ATR
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atrPeriod, adjust=False).mean()

    # Bullish and bearish bars
    bullishBar = (close - low) > (high - close)
    bearishBar = (high - close) > (close - low)

    # Time window
    # Convert time to hour in UTC
    hours = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).hour)
    inLondon = (hours >= 7) & (hours < 10)
    inNYAM = (hours >= 13) & (hours < 16)
    inNYPM = (hours >= 19) & (hours < 21)
    in_time_window = inLondon | inNYAM | inNYPM

    # Condition series
    longCondition = uptrend & volConfirmed & trendValid & macdBullish & bullishBar & (close > ema50) & in_time_window
    shortCondition = downtrend & volConfirmed & trendValid & macdBearish & bearishBar & (close < ema50) & in_time_window

    # We also need to limit by maxTradesPerDay and track trades per day.
    # Get date for each bar
    dates = df['time'].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc).date())

    # Initialize
    trades_per_day = {}
    trade_num = 1
    entries = []

    # Iterate over bars
    for i, row in df.iterrows():
        # Skip if any required indicator is NaN
        if pd.isna(ema50.iloc[i]) or pd.isna(ema200.iloc[i]) or pd.isna(volMA.iloc[i]) or pd.isna(atr.iloc[i]):
            continue

        date = dates.iloc[i]
        trades_today = trades_per_day.get(date, 0)

        # Check long condition
        if longCondition.iloc[i] and trades_today < maxTradesPerDay:
            entry_ts = row['time']
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = row['close']
            entries.append({
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
            trades_per_day[date] = trades_today + 1

        # Check short condition
        elif shortCondition.iloc[i] and trades_today < maxTradesPerDay:
            entry_ts = row['time']
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = row['close']
            entries.append({
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
            trades_per_day[date] = trades_today + 1

    return entries