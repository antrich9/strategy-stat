import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    Convert ENTRY LOGIC ONLY from Pine Script strategy.
    """
    # Default input parameters from Pine Script
    alma_window = 9
    alma_offset = 0.85
    alma_sigma = 6.0
    flux_length = 50
    flux_smooth = 1
    flux_sigma = 6
    flux_offset = 0.85
    qs_period = 20
    qs_deviation = 2.0
    vol_length = 50
    vol_normal = 75
    qfn_cooldown = 3
    use_vol_filter = True
    use_alma_signal = True
    use_flux_signal = True
    use_qs_signal = True
    
    close = df['close']
    
    # ALMA calculation function (Arnaud Legoux Moving Average)
    def calc_alma(series, window, offset, sigma):
        result = pd.Series(np.nan, index=series.index)
        for i in range(window - 1, len(series)):
            k = np.arange(window)
            weights = np.exp(-np.square(k - offset * (window - 1)) / (2 * sigma ** 2))
            weights = weights / np.sum(weights)
            result.iloc[i] = np.sum(weights * series.iloc[i - window + 1:i + 1].values)
        return result
    
    # ALMA baseline
    alma_line = calc_alma(close, alma_window, alma_offset, alma_sigma)
    
    # FluxWave oscillator
    flux_src = close.ewm(span=flux_smooth, adjust=False).mean()
    flux_alma = calc_alma(flux_src, flux_length, flux_offset, flux_sigma)
    flux_value = flux_src - flux_alma
    flux_band_mult = 1.0
    flux_band = flux_alma.rolling(flux_length).std() * flux_band_mult
    
    # QuickSilver Bands
    qs_ma = close.rolling(qs_period).mean()
    qs_std = close.rolling(qs_period).std()
    qs_upper = qs_ma + qs_deviation * qs_std
    qs_lower = qs_ma - qs_deviation * qs_std
    
    # Volume Energy
    vol_ema = df['volume'].ewm(span=vol_length, adjust=False).mean()
    vol_energy = (df['volume'] / vol_ema) * 100
    
    # HTF filter approximation (Daily on 4H chart: ~6 bars per day)
    htf_alma_approx = close.rolling(6).max().shift(-5)
    htf_long_pass = close > htf_alma_approx
    htf_short_pass = close < htf_alma_approx
    
    # Entry conditions - initialize as all False
    long_condition = pd.Series(False, index=df.index)
    short_condition = pd.Series(False, index=df.index)
    
    # ALMA Crossover signal
    if use_alma_signal:
        alma_crossup = (close > alma_line) & (close.shift(1) <= alma_line.shift(1))
        alma_crossdown = (close < alma_line) & (close.shift(1) >= alma_line.shift(1))
        long_condition = long_condition | (alma_crossup & htf_long_pass)
        short_condition = short_condition | (alma_crossdown & htf_short_pass)
    
    # FluxWave signal
    if use_flux_signal:
        flux_crossup = (flux_value > flux_band) & (flux_value.shift(1) <= flux_band.shift(1))
        flux_crossdown = (flux_value < -flux_band) & (flux_value.shift(1) >= -flux_band.shift(1))
        long_condition = long_condition | (flux_crossup & htf_long_pass)
        short_condition = short_condition | (flux_crossdown & htf_short_pass)
    
    # QuickSilver signal
    if use_qs_signal:
        qs_crossup = (close > qs_upper) & (close.shift(1) <= qs_upper.shift(1))
        qs_crossdown = (close < qs_lower) & (close.shift(1) >= qs_lower.shift(1))
        long_condition = long_condition | (qs_crossup & htf_long_pass)
        short_condition = short_condition | (qs_crossdown & htf_short_pass)
    
    # Volume filter
    if use_vol_filter:
        vol_pass = vol_energy > vol_normal
        long_condition = long_condition & vol_pass
        short_condition = short_condition & vol_pass
    
    # Build entries list
    entries = []
    trade_num = 1
    cooldown_remaining = 0
    
    for i in range(len(df)):
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue
        
        if long_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            cooldown_remaining = qfn_cooldown
        elif short_condition.iloc[i]:
            ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': df['close'].iloc[i],
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': df['close'].iloc[i],
                'raw_price_b': df['close'].iloc[i]
            })
            trade_num += 1
            cooldown_remaining = qfn_cooldown
    
    return entries