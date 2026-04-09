def generate_entries(df: pd.DataFrame) -> list:
    # default parameters
    PDCM = 70
    CDBA = 100
    FDB = 1.3
    ma_len = 20
    ma_len_b = 8
    # mode
    modo_tipo = "CON FILTRADO DE TENDENCIA"
    VVEV = True
    VVER = True
    # trend config (default)
    config_tend_alc = "DIRECCION MEDIA RAPIDA ALCISTA"
    config_tend_baj = "DIRECCION MEDIA RAPIDA BAJISTA"
    # compute indicators
    body_ratio = (df['close'] - df['open']).abs() * 100 / (df['high'] - df['low']).replace(0, np.nan)
    VVE_0 = df['close'] > df['open']
    VRE_0 = df['close'] < df['open']
    VVE_1 = VVE_0 & (body_ratio >= PDCM)
    VRE_1 = VRE_0 & (body_ratio >= PDCM)

    # ATR
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(np.abs(df['high'] - df['close'].shift(1)),
                               np.abs(df['low'] - df['close'].shift(1))))
    atr = pd.Series(tr).ewm(alpha=1/CDBA, adjust=False).mean()
    atr_prev = atr.shift(1)

    body_size = (df['close'] - df['open']).abs()
    VVE_2 = VVE_1 & (body_size >= atr_prev * FDB)
    VRE_2 = VRE_1 & (body_size >= atr_prev * FDB)

    # moving averages
    ma_slow = df['close'].rolling(window=ma_len).mean()
    ma_fast = df['close'].rolling(window=ma_len_b).mean()

    # direction slow
    diff_slow = ma_slow.diff()
    direction = pd.Series(0, index=df.index)
    direction.iloc[0] = 0
    for i in range(1, len(df)):
        if pd.isna(diff_slow.iloc[i]):
            direction.iloc[i] = direction.iloc[i-1]
        elif diff_slow.iloc[i] > 0:
            direction.iloc[i] = 1
        elif diff_slow.iloc[i] < 0:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]

    # direction fast
    diff_fast = ma_fast.diff()
    direction_b = pd.Series(0, index=df.index)
    direction_b.iloc[0] = 0
    for i in range(1, len(df)):
        if pd.isna(diff_fast.iloc[i]):
            direction_b.iloc[i] = direction_b.iloc[i-1]
        elif diff_fast.iloc[i] > 0:
            direction_b.iloc[i] = 1
        elif diff_fast.iloc[i] < 0:
            direction_b.iloc[i] = -1
        else:
            direction_b.iloc[i] = direction_b.iloc[i-1]

    # trend conditions
    # Only need DMRA (direction fast > 0) and DMRB (direction fast < 0)
    DMRA = direction_b > 0
    DMRB = direction_b < 0

    VVE_3 = VVE_2 & DMRA
    VRE_3 = VRE_2 & DMRB

    # final signals
    RES_VVE = VVE_3  # VVEV true and modo_tipo == "CON FILTRADO DE TENDENCIA"
    RES_VRE = VRE_3

    entries = []
    trade_num = 1
    for i in range(len(df)):
        if RES_VVE.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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
        elif RES_VRE.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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
    return entries