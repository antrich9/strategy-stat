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
    # Parameters (from input defaults)
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
    use_alma_signal = True
    use_flux_signal = True
    use_qs_signal = True
    use_vol_filter = True
    entry_threshold = 0.5

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # ALMA - Arnaud Legoux Moving Average
    def alma_calculate(src, window, offset, sigma):
        result = np.full(len(src), np.nan)
        m = offset * (window - 1)
        s = sigma * (window - 1) / 6
        for i in range(window - 1, len(src)):
            wSum = 0.0
            wTs = 0.0
            for n in range(window):
                w = np.exp(-((n - m) ** 2) / (2 * s * s))
                wSum += w * src.iloc[i - window + 1 + n]
                wTs += w
            if wTs != 0:
                result[i] = wSum / wTs
        return pd.Series(result, index=src.index)

    # SMA for QuickSilver
    sma_qs = close.rolling(qs_period).mean()
    std_qs = close.rolling(qs_period).std()
    qs_upper = sma_qs + qs_deviation * std_qs
    qs_lower = sma_qs - qs_deviation * std_qs

    # ALMA baseline
    alma = alma_calculate(close, alma_window, alma_offset, alma_sigma)

    # FluxWave Oscillator (simplified trend identifier)
    ema_flux = close.ewm(span=flux_length, adjust=False).mean()
    flux_wave = close - ema_flux
    flux_wave_smooth = flux_wave.ewm(span=flux_smooth, adjust=False).mean()
    flux_band_mult = 1.0
    flux_upper_band = flux_wave_smooth.rolling(flux_length).std() * flux_band_mult
    flux_lower_band = -flux_wave_smooth.rolling(flux_length).std() * flux_band_mult

    # Volume Energy Filter (normalized)
    vol_sma = volume.rolling(vol_length).mean()
    vol_std = volume.rolling(vol_length).std()
    vol_energy = ((volume - vol_sma) / (vol_std + 1e-9)) * 100
    vol_filter_ok = vol_energy > 0

    # Build conditions
    alma_crossover_long = (alma > alma.shift(1)) & (alma.shift(1) <= alma.shift(2))
    alma_crossunder_short = (alma < alma.shift(1)) & (alma.shift(1) >= alma.shift(2))

    flux_oversold = flux_wave_smooth < flux_lower_band
    flux_overbought = flux_wave_smooth > flux_upper_band

    price_above_alma = close > alma
    price_below_alma = close < alma

    price_above_qs = close > qs_upper
    price_below_qs = close < qs_lower

    # Entry conditions
    long_condition = pd.Series(False, index=df.index)
    short_condition = pd.Series(False, index=df.index)

    if use_alma_signal and use_flux_signal and use_qs_signal and use_vol_filter:
        long_condition = alma_crossover_long & price_above_alma & flux_oversold & price_above_qs & vol_filter_ok
        short_condition = alma_crossunder_short & price_below_alma & flux_overbought & price_below_qs & vol_filter_ok
    elif use_alma_signal and use_flux_signal and use_vol_filter:
        long_condition = alma_crossover_long & price_above_alma & flux_oversold & vol_filter_ok
        short_condition = alma_crossunder_short & price_below_alma & flux_overbought & vol_filter_ok
    elif use_alma_signal and use_qs_signal and use_vol_filter:
        long_condition = alma_crossover_long & price_above_alma & price_above_qs & vol_filter_ok
        short_condition = alma_crossunder_short & price_below_alma & price_below_qs & vol_filter_ok
    elif use_flux_signal and use_qs_signal and use_vol_filter:
        long_condition = flux_oversold & price_above_qs & vol_filter_ok
        short_condition = flux_overbought & price_below_qs & vol_filter_ok
    elif use_alma_signal and use_vol_filter:
        long_condition = alma_crossover_long & price_above_alma & vol_filter_ok
        short_condition = alma_crossunder_short & price_below_alma & vol_filter_ok
    elif use_flux_signal and use_vol_filter:
        long_condition = flux_oversold & vol_filter_ok
        short_condition = flux_overbought & vol_filter_ok
    elif use_qs_signal and use_vol_filter:
        long_condition = price_above_qs & vol_filter_ok
        short_condition = price_below_qs & vol_filter_ok
    elif use_alma_signal and use_flux_signal:
        long_condition = alma_crossover_long & price_above_alma & flux_oversold
        short_condition = alma_crossunder_short & price_below_alma & flux_overbought
    elif use_alma_signal and use_qs_signal:
        long_condition = alma_crossover_long & price_above_alma & price_above_qs
        short_condition = alma_crossunder_short & price_below_alma & price_below_qs
    elif use_flux_signal and use_qs_signal:
        long_condition = flux_oversold & price_above_qs
        short_condition = flux_overbought & price_below_qs
    elif use_alma_signal:
        long_condition = alma_crossover_long & price_above_alma
        short_condition = alma_crossunder_short & price_below_alma
    elif use_flux_signal:
        long_condition = flux_oversold
        short_condition = flux_overbought
    elif use_qs_signal:
        long_condition = price_above_qs
        short_condition = price_below_qs
    elif use_vol_filter:
        long_condition = vol_filter_ok
        short_condition = vol_filter_ok

    long_condition = long_condition.fillna(False)
    short_condition = short_condition.fillna(False)

    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(alma.iloc[i]) or pd.isna(flux_wave_smooth.iloc[i]) or pd.isna(qs_upper.iloc[i]):
            continue

        direction = None
        if long_condition.iloc[i]:
            direction = 'long'
        elif short_condition.iloc[i]:
            direction = 'short'

        if direction:
            entry_price = close.iloc[i]
            ts = int(df['time'].iloc[i])
            entry_time_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': float(entry_price),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(entry_price),
                'raw_price_b': float(entry_price)
            })
            trade_num += 1

    return entries