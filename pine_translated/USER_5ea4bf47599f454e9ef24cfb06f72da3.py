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
    # Strategy parameters (default values from Pine Script inputs)
    alma_window = 9
    alma_offset = 0.85
    alma_sigma = 6.0
    flux_length = 50
    flux_smooth = 1
    flux_offset = 0.85
    flux_sigma = 6
    qs_period = 20
    qs_deviation = 2.0
    vol_length = 50
    vol_high = 150
    vol_normal = 75
    entry_lookback = 5
    entry_threshold = 0.5
    entry_mode = "Combined"
    use_alma_signal = True
    use_flux_signal = True
    use_qs_signal = True
    use_vol_filter = True
    flux_signal_mode = "OS/OB"
    flux_band_mult = 1.0
    qfn_cooldown = 3

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # ALMA Implementation
    def alma(src, window, offset, sigma):
        m = offset * (window - 1)
        s = window / sigma
        w = np.arange(window)
        w = np.exp(-((w - m) ** 2) / (2 * s * s))
        w = w / w.sum()
        result = np.zeros(len(src))
        for i in range(window - 1, len(src)):
            result[i] = np.sum(w * src[i - window + 1:i + 1])
        return pd.Series(result, index=src.index)

    # FluxWave Calculation
    def fluxwave(src, length, smooth, alma_offset_val, alma_sigma_val):
        ema_fast = src.ewm(span=12, adjust=False).mean()
        ema_slow = src.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_base = alma(macd, length, alma_offset_val, alma_sigma_val)
        for _ in range(smooth - 1):
            signal_base = alma(signal_base, length, alma_offset_val, alma_sigma_val)
        upper = alma(signal_base, length, alma_offset_val, alma_sigma_val) + flux_band_mult * signal_base.std()
        lower = alma(signal_base, length, alma_offset_val, alma_sigma_val) - flux_band_mult * signal_base.std()
        return signal_base, upper, lower

    # QuickSilver Bands
    qs_upper = close.rolling(qs_period).mean() + qs_deviation * close.rolling(qs_period).std()
    qs_middle = close.rolling(qs_period).mean()
    qs_lower = close.rolling(qs_period).mean() - qs_deviation * close.rolling(qs_period).std()

    # Volume Energy (Wilder Smoothed)
    vol_ma = volume.rolling(vol_length).mean()
    vol_std = volume.rolling(vol_length).std()
    vol_energy_raw = ((volume - vol_ma) / (vol_std + 1e-9)) * 100
    vol_energy = vol_energy_raw.ewm(alpha=1.0/vol_length, adjust=False).mean()

    # ALMA Baseline
    alma_line = alma(close, alma_window, alma_offset, alma_sigma)
    alma_trend = alma(close, alma_window * 2, alma_offset, alma_sigma)

    # FluxWave
    flux_signal, flux_upper, flux_lower = fluxwave(close, flux_length, flux_smooth, flux_offset, flux_sigma)

    # Build conditions
    alma_long_cond = (close > alma_line) & (alma_line > alma_trend)
    alma_short_cond = (close < alma_line) & (alma_line < alma_trend)

    if flux_signal_mode == "OS/OB" or flux_signal_mode == "OS/OB enter":
        flux_long_cond = flux_signal < flux_lower
        flux_short_cond = flux_signal > flux_upper
    elif flux_signal_mode == "OS/OB exit":
        flux_long_cond = flux_signal > flux_lower
        flux_short_cond = flux_signal < flux_upper
    elif flux_signal_mode == "OS/OB reverse":
        flux_long_cond = flux_signal > flux_upper
        flux_short_cond = flux_signal < flux_lower
    elif flux_signal_mode == "OS/OB enter reverse":
        flux_long_cond = flux_signal > flux_upper
        flux_short_cond = flux_signal < flux_lower
    else:
        flux_long_cond = (flux_signal > flux_lower) & (flux_signal.shift(1) <= flux_lower)
        flux_short_cond = (flux_signal < flux_upper) & (flux_signal.shift(1) >= flux_upper)

    qs_long_cond = close > qs_upper
    qs_short_cond = close < qs_lower

    vol_filter_pass = vol_energy > vol_normal

    # Cooldown tracking
    cooldown_counter = pd.Series(0, index=df.index)

    entries = []
    trade_num = 1
    last_long_bar = -qfn_cooldown
    last_short_bar = -qfn_cooldown

    for i in range(1, len(df)):
        if pd.isna(alma_line.iloc[i]) or pd.isna(flux_signal.iloc[i]) or pd.isna(qs_upper.iloc[i]) or pd.isna(vol_energy.iloc[i]):
            continue

        if entry_mode == "ALMA":
            long_cond = alma_long_cond.iloc[i]
            short_cond = alma_short_cond.iloc[i]
        elif entry_mode == "FluxWave":
            long_cond = flux_long_cond.iloc[i]
            short_cond = flux_short_cond.iloc[i]
        elif entry_mode == "QuickSilver":
            long_cond = qs_long_cond.iloc[i]
            short_cond = qs_short_cond.iloc[i]
        else:  # Combined or Custom
            if use_alma_signal and use_flux_signal and use_qs_signal:
                long_cond = alma_long_cond.iloc[i] and flux_long_cond.iloc[i] and qs_long_cond.iloc[i]
                short_cond = alma_short_cond.iloc[i] and flux_short_cond.iloc[i] and qs_short_cond.iloc[i]
            elif use_alma_signal and use_flux_signal:
                long_cond = alma_long_cond.iloc[i] and flux_long_cond.iloc[i]
                short_cond = alma_short_cond.iloc[i] and flux_short_cond.iloc[i]
            elif use_alma_signal and use_qs_signal:
                long_cond = alma_long_cond.iloc[i] and qs_long_cond.iloc[i]
                short_cond = alma_short_cond.iloc[i] and qs_short_cond.iloc[i]
            elif use_flux_signal and use_qs_signal:
                long_cond = flux_long_cond.iloc[i] and qs_long_cond.iloc[i]
                short_cond = flux_short_cond.iloc[i] and qs_short_cond.iloc[i]
            elif use_alma_signal:
                long_cond = alma_long_cond.iloc[i]
                short_cond = alma_short_cond.iloc[i]
            elif use_flux_signal:
                long_cond = flux_long_cond.iloc[i]
                short_cond = flux_short_cond.iloc[i]
            elif use_qs_signal:
                long_cond = qs_long_cond.iloc[i]
                short_cond = qs_short_cond.iloc[i]
            else:
                long_cond = False
                short_cond = False

        if use_vol_filter:
            long_cond = long_cond and vol_filter_pass.iloc[i]
            short_cond = short_cond and vol_filter_pass.iloc[i]

        # Entry lookback confirmation
        if entry_lookback > 1:
            long_lookback = long_cond
            short_lookback = short_cond
            for j in range(1, min(entry_lookback, i + 1)):
                if i - j >= 0:
                    if entry_mode == "ALMA":
                        long_lookback = long_lookback and alma_long_cond.iloc[i - j]
                        short_lookback = short_lookback and alma_short_cond.iloc[i - j]
                    elif entry_mode == "FluxWave":
                        long_lookback = long_lookback and flux_long_cond.iloc[i - j]
                        short_lookback = short_lookback and flux_short_cond.iloc[i - j]
                    elif entry_mode == "QuickSilver":
                        long_lookback = long_lookback and qs_long_cond.iloc[i - j]
                        short_lookback = short_lookback and qs_short_cond.iloc[i - j]
                    else:
                        if use_alma_signal and use_flux_signal and use_qs_signal:
                            long_lookback = long_lookback and alma_long_cond.iloc[i - j] and flux_long_cond.iloc[i - j] and qs_long_cond.iloc[i - j]
                            short_lookback = short_lookback and alma_short_cond.iloc[i - j] and flux_short_cond.iloc[i - j] and qs_short_cond.iloc[i - j]
                        else:
                            long_lookback = long_lookback and flux_long_cond.iloc[i - j] if use_flux_signal else long_lookback and qs_long_cond.iloc[i - j] if use_qs_signal else long_lookback
                            short_lookback = short_lookback and flux_short_cond.iloc[i - j] if use_flux_signal else short_lookback and qs_short_cond.iloc[i - j] if use_qs_signal else short_lookback
            long_cond = long_cond and long_lookback
            short_cond = short_cond and short_lookback

        entry_price = close.iloc[i]
        entry_ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat() if entry_ts > 1e12 else datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()

        if long_cond and (i - last_long_bar) > qfn_cooldown:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            last_long_bar = i
            trade_num += 1

        if short_cond and (i - last_short_bar) > qfn_cooldown:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            last_short_bar = i
            trade_num += 1

    return entries