def generate_entries(df: pd.DataFrame) -> list:
    # Ensure sorted
    df = df.sort_values('time').reset_index(drop=True)
    
    # Basic price arrays
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # is_up, is_down
    is_up = close > open_
    is_down = close < open_
    
    # Indicators
    # Volume filter
    sma_vol9 = volume.rolling(9).mean()
    vol_filter = (volume.shift(1) > sma_vol9 * 1.5)
    
    # ATR (Wilder)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/20, adjust=False).mean()
    
    # ATR filter
    atr_filter = ((low - high.shift(2) > atr) | (low.shift(2) - high > atr))
    
    # Trend filter (SMA 54)
    sma_loc = close.rolling(54).mean()
    loc2 = sma_loc > sma_loc.shift(1)
    locfiltb = loc2
    locfilts = ~loc2
    
    # FVG (bullish)
    bfvg = (low > high.shift(2)) & vol_filter & atr_filter & locfiltb
    # FVG (bearish)
    sfvg = (high < low.shift(2)) & vol_filter & atr_filter & locfilts
    
    # Stacked OB+FVG
    # obUp: down 2 bars ago, up 1 bar ago, close of up bar > high of down bar
    obUp = is_down.shift(2) & is_up.shift(1) & (close.shift(1) > high.shift(2))
    fvgUp = low > high.shift(2)
    stacked_up = obUp & fvgUp
    
    obDown = is_up.shift(2) & is_down.shift(1) & (close.shift(1) < low.shift(2))
    fvgDown = high < low.shift(2)
    stacked_down = obDown & fvgDown
    
    # Time window
    ts = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Europe/London')
    hour = ts.dt.hour
    minute = ts.dt.minute
    time_in_min = hour * 60 + minute
    in_morning = (time_in_min >= 7*60+45) & (time_in_min < 9*60+45)
    in_afternoon = (time_in_min >= 14*60+45) & (time_in_min < 16*60+45)
    in_window = in_morning | in_afternoon
    
    # Entry signals
    long_cond = (stacked_up | bfvg) & in_window
    short_cond = (stacked_down | sfvg) & in_window
    
    # Build entries
    entries = []
    trade_num = 1
    for i in df.index:
        if long_cond.iloc[i]:
            entry = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            }
            entries.append(entry)
            trade_num += 1
        elif short_cond.iloc[i]:
            entry = {
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            }
            entries.append(entry)
            trade_num += 1
    return entries