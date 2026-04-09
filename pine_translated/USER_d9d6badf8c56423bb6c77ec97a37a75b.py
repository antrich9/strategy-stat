def generate_entries(df: pd.DataFrame) -> list:
    # Ensure time column is datetime
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.sort_values('time').reset_index(drop=True)

    # Resample to 4H
    # Use the start of each 4h period as the key
    df['4h_start'] = df['time'].dt.floor('4h')
    # For each group, get open (first), high (max), low (min), close (last), volume (sum)
    # Also get the timestamp of the last row in group (close_time)
    agg = df.groupby('4h_start').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
        close_time=('time', 'last')
    ).reset_index(drop=True)

    # Now agg has columns: open, high, low, close, volume, close_time
    # The index corresponds to each 4H bar.

    # Compute indicators
    # Volume SMA(9)
    agg['vol_sma9'] = agg['volume'].rolling(window=9, min_periods=9).mean()
    # Volume filter: previous volume > sma*1.5
    agg['volfilt'] = True  # default true (disabled)
    # If we want to enable: agg['volume'].shift(1) > agg['vol_sma9'] * 1.5

    # Compute ATR (Wilder) length 20
    def wilder_atr(df_high, df_low, df_close, length=20):
        tr = pd.concat([df_high - df_low,
                        (df_high - df_close.shift(1)).abs(),
                        (df_low - df_close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=length, min_periods=length).mean()
        # Wilder smoothing: for i > length: atr = (prev_atr * (length-1) + tr) / length
        for i in range(length, len(tr)):
            atr.iloc[i] = (atr.iloc[i-1] * (length - 1) + tr.iloc[i]) / length
        return atr

    agg['atr'] = wilder_atr(agg['high'], agg['low'], agg['close'], 20)
    # ATR filter: (low - high[2] > atr) or (low[2] - high > atr)
    # We'll compute shift(2) for high and low
    agg['low_shift2'] = agg['low'].shift(2)
    agg['high_shift2'] = agg['high'].shift(2)
    # Bullish gap condition: low - high_shift2 > atr
    agg['bull_gap'] = agg['low'] - agg['high_shift2'] > agg['atr']
    # Bearish gap condition: low_shift2 - high > atr
    agg['bear_gap'] = agg['low_shift2'] - agg['high'] > agg['atr']
    agg['atrfilt'] = True  # default disabled

    # Trend filter: SMA 54 of close
    agg['sma54'] = agg['close'].rolling(window=54, min_periods=54).mean()
    agg['trend_up'] = agg['sma54'] > agg['sma54'].shift(1)
    # Default disabled, but we can compute anyway
    agg['locfiltb'] = True  # if inp31 else trend_up
    agg['locfilts'] = True  # if inp31 else ~trend_up

    # FVG conditions
    # Bullish: low > high[2] and volfilt and atrfilt and locfiltb
    # We need high[2] from 2 bars ago: shift(2)
    agg['high_shift2'] = agg['high'].shift(2)
    agg['bfvg'] = (agg['low'] > agg['high_shift2']) & agg['volfilt'] & agg['atrfilt'] & agg['locfiltb']

    # Bearish: high < low[2] and volfilt and atrfilt and locfilts
    agg['low_shift2'] = agg['low'].shift(2)
    agg['sfvg'] = (agg['high'] < agg['low_shift2']) & agg['volfilt'] & agg['atrfilt'] & agg['locfilts']

    # Sharp turn detection
    lastFVG = 0
    trade_num = 1
    entries = []

    for i in range(len(agg)):
        bfvg = agg['bfvg'].iloc[i] if not pd.isna(agg['bfvg'].iloc[i]) else False
        sfvg = agg['sfvg'].iloc[i] if not pd.isna(agg['sfvg'].iloc[i]) else False

        # Sharp turn conditions
        if bfvg and lastFVG == -1:
            direction = 'long'
            entry_price = agg['close'].iloc[i]
            entry_ts = int(agg['close_time'].iloc[i].timestamp())
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
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
        elif sfvg and lastFVG == 1:
            direction = 'short'
            entry_price = agg['close'].iloc[i]
            entry_ts = int(agg['close_time'].iloc[i].timestamp())
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
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

        # Update lastFVG
        if bfvg:
            lastFVG = 1
        elif sfvg:
            lastFVG = -1

    return entries