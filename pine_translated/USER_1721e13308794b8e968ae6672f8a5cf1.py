import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # ALMA function
    def alma(src, length, offset, sigma):
        w = np.exp(-np.square(np.arange(length) - offset * (length - 1)) / (2 * sigma * sigma))
        w /= w.sum()
        return pd.Series(np.convolve(src, w, mode='same'), index=src.index)
    
    # Jurik MA (simplified ALMA variant)
    jma_len = 14
    jma_phase = 0
    jma_power = 2.0
    offset = (jma_phase + 100) / 100
    sigma = jma_len / (jma_len + jma_power * 30)
    alma_raw = alma(close.values, jma_len, offset, sigma)
    jma = pd.Series(alma_raw, index=close.index)
    
    # RSI (Wilder)
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/jma_len, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/jma_len, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # ATR (Wilder)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # QuickSilver Bands
    qs_period = 20
    qs_deviation = 2.0
    qs_sma = close.rolling(qs_period).mean()
    qs_std = close.rolling(qs_period).std()
    qs_upper = qs_sma + qs_std * qs_deviation
    qs_lower = qs_sma - qs_std * qs_deviation
    
    # FluxWave (approximation of advanced oscillator)
    flux_smooth = 1
    flux_length = 50
    flux_offset = 0.85
    flux_sigma = 6
    flux_band_mult = 1.0
    
    flux_ewm = close.ewm(span=flux_smooth, adjust=False).mean()
    flux_vol_ratio = close / (volume / volume.rolling(flux_length).mean() + 1)
    flux_alma = alma(flux_vol_ratio.values, flux_length, flux_offset, flux_sigma)
    flux_wave = pd.Series(flux_alma, index=close.index).ewm(span=flux_smooth, adjust=False).mean()
    flux_std = flux_wave.rolling(flux_length).std()
    flux_upper_band = flux_wave + flux_std * flux_band_mult
    flux_lower_band = flux_wave - flux_std * flux_band_mult
    
    # Volume Energy
    vol_length = 50
    vol_ma = volume.rolling(vol_length).mean()
    vol_energy = (volume / vol_ma) * 100
    vol_threshold = 75.0
    
    # Entry signals
    entry_lookback = 5
    use_vol_filter = True
    use_alma_signal = True
    use_flux_signal = True
    use_qs_signal = True
    
    # Price vs JMA
    price_above_jma = close > jma
    price_below_jma = close < jma
    price_cross_above_jma = (close > jma) & (close.shift(1) <= jma.shift(1))
    price_cross_below_jma = (close < jma) & (close.shift(1) >= jma.shift(1))
    
    # FluxWave signals (oscillator crossover mode)
    fluxbull = flux_wave > flux_wave.shift(1)
    fluxbear = flux_wave < flux_wave.shift(1)
    fluxbull_signal = (flux_wave > flux_lower_band) & fluxbull
    fluxbear_signal = (flux_wave < flux_upper_band) & fluxbear
    
    # QuickSilver breakout
    qs_bull_break = close > qs_upper
    qs_bear_break = close < qs_lower
    
    # Volume confirmation
    vol_confirm = vol_energy > vol_threshold
    
    # Lookback validation
    def check_lookback(condition_series, lookback):
        result = pd.Series(True, index=condition_series.index)
        for i in range(1, lookback + 1):
            shifted = condition_series.shift(i)
            result = result & shifted.fillna(False)
        return result
    
    bull_lookback = check_lookback(price_above_jma, entry_lookback)
    bear_lookback = check_lookback(price_below_jma, entry_lookback)
    
    # Combined entry conditions
    long_cond = price_cross_above_jma & bull_lookback
    if use_flux_signal:
        long_cond = long_cond & fluxbull_signal
    if use_qs_signal:
        long_cond = long_cond & qs_bull_break
    if use_vol_filter:
        long_cond = long_cond & vol_confirm
    
    short_cond = price_cross_below_jma & bear_lookback
    if use_flux_signal:
        short_cond = short_cond & fluxbear_signal
    if use_qs_signal:
        short_cond = short_cond & qs_bear_break
    if use_vol_filter:
        short_cond = short_cond & vol_confirm
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(jma.iloc[i]) or pd.isna(flux_wave.iloc[i]):
            continue
        
        direction = None
        if long_cond.iloc[i]:
            direction = 'long'
        elif short_cond.iloc[i]:
            direction = 'short'
        
        if direction:
            entry_ts = int(df['time'].iloc[i])
            entry_price = float(close.iloc[i])
            
            entries.append({
                'trade_num': trade_num,
                'direction': direction,
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
    
    return entries