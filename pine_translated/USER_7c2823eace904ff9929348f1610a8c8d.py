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
    # Strategy parameters (from input defaults)
    alma_window = 9
    alma_offset = 0.85
    alma_sigma = 6.0
    flux_length = 50
    flux_smooth = 1
    flux_offset = 0.85
    flux_sigma = 6
    flux_band_mult = 1.0
    flux_signal_mode = "OS/OB"
    qs_period = 20
    qs_deviation = 2.0
    vol_length = 50
    vol_high = 150
    vol_normal = 75
    vol_low = 75
    vol_filter_signals = True
    entry_lookback = 5
    entry_threshold = 0.5
    qfn_cooldown = 3
    use_alma_signal = True
    use_flux_signal = True
    use_qs_signal = True
    use_vol_filter = True

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # ALMA calculation
    def calc_alma(src, window, offset, sigma):
        m = offset * (window - 1)
        s = sigma * window
        k = np.arange(window)
        w = np.exp(-((k - m) ** 2) / (2 * s ** 2))
        w = w / w.sum()
        alma = np.convolve(src, w, mode='same')
        return pd.Series(alma, index=src.index)

    alma = calc_alma(close, alma_window, alma_offset, alma_sigma)

    # FluxWave calculation
    def calc_alma_flux(src, window, offset, sigma):
        m = offset * (window - 1)
        s = sigma * window
        k = np.arange(window)
        w = np.exp(-((k - m) ** 2) / (2 * s ** 2))
        w = w / w.sum()
        alma_vals = np.convolve(src, w, mode='same')
        return pd.Series(alma_vals, index=src.index)

    mom = close.diff(flux_length)
    mom_ma = mom.ewm(span=flux_smooth, adjust=False).mean()
    flux_wave = mom_ma
    flux_smoothed = calc_alma_flux(flux_wave, flux_length, flux_offset, flux_sigma)
    flux_signal = flux_smoothed.diff()
    flux_signal_smoothed = flux_signal.ewm(span=flux_smooth, adjust=False).mean()

    flux_std = flux_signal_smoothed.rolling(flux_length).std()
    flux_upper = flux_std * flux_band_mult
    flux_lower = -flux_std * flux_band_mult

    # QuickSilver Bands calculation
    qs_ma = close.rolling(qs_period).mean()
    qs_std = close.rolling(qs_period).std()
    qs_upper = qs_ma + qs_std * qs_deviation
    qs_lower = qs_ma - qs_std * qs_deviation

    # Volume Energy calculation
    vol_sma = volume.rolling(vol_length).mean()
    vol_energy = (volume / vol_sma) * 100
    vol_high_energy = vol_energy > vol_high

    # Entry conditions
    alma_crossover_long = (close > alma) & (close.shift(1) <= alma.shift(1))
    alma_crossunder_short = (close < alma) & (close.shift(1) >= alma.shift(1))

    flux_long_cond = flux_signal_smoothed > flux_lower
    flux_short_cond = flux_signal_smoothed < flux_upper

    qs_breakout_long = close > qs_upper
    qs_breakout_short = close < qs_lower

    vol_ok = vol_energy >= vol_normal if vol_filter_signals else True

    # Combined entry signals
    if use_alma_signal and use_flux_signal and use_qs_signal:
        long_cond = alma_crossover_long & flux_long_cond & qs_breakout_long & vol_ok
        short_cond = alma_crossunder_short & flux_short_cond & qs_breakout_short & vol_ok
    elif use_alma_signal and use_flux_signal:
        long_cond = alma_crossover_long & flux_long_cond & vol_ok
        short_cond = alma_crossunder_short & flux_short_cond & vol_ok
    elif use_alma_signal and use_qs_signal:
        long_cond = alma_crossover_long & qs_breakout_long & vol_ok
        short_cond = alma_crossunder_short & qs_breakout_short & vol_ok
    elif use_flux_signal and use_qs_signal:
        long_cond = flux_long_cond & qs_breakout_long & vol_ok
        short_cond = flux_short_cond & qs_breakout_short & vol_ok
    elif use_alma_signal:
        long_cond = alma_crossover_long & vol_ok
        short_cond = alma_crossunder_short & vol_ok
    elif use_flux_signal:
        long_cond = flux_long_cond & vol_ok
        short_cond = flux_short_cond & vol_ok
    elif use_qs_signal:
        long_cond = qs_breakout_long & vol_ok
        short_cond = qs_breakout_short & vol_ok
    else:
        long_cond = pd.Series(False, index=df.index)
        short_cond = pd.Series(False, index=df.index)

    # Confirmation bars check
    long_confirmed = long_cond.rolling(entry_lookback).sum() >= entry_lookback
    short_confirmed = short_cond.rolling(entry_lookback).sum() >= entry_lookback

    # Build entry list
    entries = []
    trade_num = 1
    last_entry_idx = -qfn_cooldown - 1

    for i in range(len(df)):
        if i <= flux_length or i <= qs_period or i <= vol_length:
            continue

        entry_signal = False
        direction = None

        if long_confirmed.iloc[i] and (i - last_entry_idx) > qfn_cooldown:
            entry_signal = True
            direction = 'long'
        elif short_confirmed.iloc[i] and (i - last_entry_idx) > qfn_cooldown:
            entry_signal = True
            direction = 'short'

        if entry_signal:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(close.iloc[i])

            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

            trade_num += 1
            last_entry_idx = i

    return entries