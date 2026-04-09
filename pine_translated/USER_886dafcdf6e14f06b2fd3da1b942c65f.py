import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # ALMA Parameters (from inputs)
    alma_window = 9
    alma_offset = 0.85
    alma_sigma = 6.0
    use_alma_signal = True
    
    # FluxWave Parameters (from inputs)
    flux_smooth = 1
    flux_length = 50
    flux_offset = 0.85
    flux_sigma = 6
    flux_band_mult = 1.0
    flux_signal_mode = "OS/OB"
    use_flux_signal = True
    
    # QuickSilver Parameters (from inputs)
    qs_period = 20
    qs_deviation = 2.0
    use_qs_signal = True
    
    # Volume Energy Parameters (from inputs)
    vol_length = 50
    use_vol_filter = True
    
    # Entry Parameters (from inputs)
    entry_mode = "Combined"
    entry_lookback = 5
    entry_threshold = 0.5
    use_flux_signal = True
    use_qs_signal = True
    use_alma_signal = True
    use_vol_filter = True
    
    # ALMA Calculation
    k = alma_window
    m = (k - 1) * alma_offset
    s = alma_sigma
    weights = np.exp(-((np.arange(k) - m) ** 2) / (2 * s ** 2))
    weights = weights / weights.sum()
    
    alma = close.rolling(window=k, min_periods=k).apply(lambda x: np.sum(weights * x), raw=True)
    
    # FluxWave Oscillator (advanced Trendilio variant)
    # FluxWave uses ALMA-based cycle oscillator with bands
    flux_alma_weights = np.exp(-((np.arange(flux_sigma * 2 + 1) - (flux_sigma * 2) * flux_offset / 2) ** 2) / (2 * ((flux_sigma * 2 + 1) / 6) ** 2))
    flux_alma_weights = flux_alma_weights / flux_alma_weights.sum()
    
    flux_src = close.rolling(window=flux_sigma * 2 + 1, min_periods=flux_sigma * 2 + 1).apply(lambda x: np.sum(flux_alma_weights * x), raw=True)
    
    flux_ema = flux_src.ewm(span=flux_smooth, adjust=False).mean()
    flux_atr = (flux_src - flux_src.shift(1)).abs().rolling(window=flux_length).mean()
    flux_std = flux_src.rolling(window=flux_length).std()
    
    flux_upper = flux_ema + flux_std * flux_band_mult
    flux_lower = flux_ema - flux_std * flux_band_mult
    flux_wave = (flux_src - flux_ema) / (flux_std.replace(0, np.nan) + 1e-10)
    
    # QuickSilver Bands (similar to Bollinger Bands with EMA base)
    qs_ema = close.ewm(span=qs_period, adjust=False).mean()
    qs_std = close.rolling(window=qs_period).std()
    qs_upper = qs_ema + qs_std * qs_deviation
    qs_lower = qs_ema - qs_std * qs_deviation
    
    # Volume Energy (Normalized Volume)
    vol_ma = volume.rolling(window=vol_length, min_periods=vol_length).mean()
    vol_std = volume.rolling(window=vol_length, min_periods=vol_length).std()
    vol_energy = (volume - vol_ma) / (vol_std.replace(0, np.nan) + 1e-10) * 100
    
    # Build condition Series
    alma_cross_up = (close > alma) & (close.shift(1) <= alma.shift(1))
    alma_cross_down = (close < alma) & (close.shift(1) >= alma.shift(1))
    
    flux_os = flux_wave < -1.0
    flux_ob = flux_wave > 1.0
    
    qs_break_up = close > qs_upper
    qs_break_down = close < qs_lower
    
    vol_confirm = vol_energy > 50
    
    # ALMA Signal for Combined mode
    alma_long_cond = alma_cross_up
    alma_short_cond = alma_cross_down
    
    # FluxWave Long (OS/OB mode - enter on OS reversal)
    flux_long_cond = (flux_os) & (flux_wave.shift(1) < flux_os.shift(1))
    flux_short_cond = (flux_ob) & (flux_wave.shift(1) > flux_ob.shift(1))
    
    # QuickSilver Long (breakout mode)
    qs_long_cond = qs_break_up
    qs_short_cond = qs_break_down
    
    # Entry strength calculation (lookback confirmation)
    def calc_strength(long_cond, short_cond, lookback):
        long_count = long_cond.rolling(window=lookback, min_periods=1).sum()
        short_count = short_cond.rolling(window=lookback, min_periods=1).sum()
        return (long_count - short_count) / lookback
    
    entry_long_strength = calc_strength(alma_long_cond | qs_break_up, alma_short_cond | qs_break_down, entry_lookback)
    entry_short_strength = calc_strength(alma_short_cond | qs_break_down, alma_long_cond | qs_break_up, entry_lookback)
    
    # Combined entry conditions based on entry_mode
    if entry_mode == "Combined":
        long_entry = alma_long_cond & (qs_break_up | flux_long_cond) & (entry_long_strength >= entry_threshold)
        short_entry = alma_short_cond & (qs_break_down | flux_short_cond) & (entry_short_strength >= entry_threshold)
    elif entry_mode == "ALMA":
        long_entry = alma_long_cond & (entry_long_strength >= entry_threshold)
        short_entry = alma_short_cond & (entry_short_strength >= entry_threshold)
    elif entry_mode == "FluxWave":
        long_entry = flux_long_cond & (entry_long_strength >= entry_threshold)
        short_entry = flux_short_cond & (entry_short_strength >= entry_threshold)
    elif entry_mode == "QuickSilver":
        long_entry = qs_break_up & (entry_long_strength >= entry_threshold)
        short_entry = qs_break_down & (entry_short_strength >= entry_threshold)
    else:
        long_entry = alma_long_cond & qs_break_up & (entry_long_strength >= entry_threshold)
        short_entry = alma_short_cond & qs_break_down & (entry_short_strength >= entry_threshold)
    
    # Apply volume filter
    if use_vol_filter:
        long_entry = long_entry & vol_confirm
        short_entry = short_entry & vol_confirm
    
    # Generate entries
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if pd.isna(alma.iloc[i]) or pd.isna(qs_upper.iloc[i]) or pd.isna(flux_wave.iloc[i]):
            continue
        
        ts = int(df['time'].iloc[i])
        price = close.iloc[i]
        
        if long_entry.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                'entry_price_guess': price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': price,
                'raw_price_b': price
            })
            trade_num += 1
    
    return entries