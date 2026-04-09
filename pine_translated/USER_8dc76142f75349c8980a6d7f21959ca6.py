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
    
    # ========== PARAMETERS ==========
    alma_window = 9
    alma_offset = 0.85
    alma_sigma = 6.0
    
    flux_smooth = 1
    flux_length = 50
    flux_offset = 0.85
    flux_sigma = 6
    flux_band_mult = 1.0
    flux_signal_mode = "OS/OB"
    
    qs_period = 20
    qs_deviation = 2.0
    
    vol_length = 50
    vol_high = 150
    vol_normal = 75
    
    use_alma_signal = True
    use_flux_signal = True
    use_qs_signal = True
    use_vol_filter = True
    
    qfn_cooldown = 3
    entry_lookback = 5
    entry_threshold = 0.5
    
    # ========== HELPER FUNCTIONS ==========
    def alma(source, window, offset, sigma):
        m = np.floor(offset * (window - 1))
        s = window / sigma
        wts = np.exp(-((np.arange(window) - m) ** 2) / (2 * s * s))
        wts = wts / wts.sum()
        
        result = np.zeros(len(source))
        for i in range(window - 1, len(source)):
            result[i] = np.sum(wts * source[i - window + 1:i + 1])
        return pd.Series(result, index=source.index)
    
    def wilder_rsi(src, length):
        delta = src.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def wilder_tr(src, high, low):
        tr = pd.concat([high - low, (high - src.shift(1)).abs(), (low - src.shift(1)).abs()], axis=1).max(axis=1)
        return tr
    
    def wilder_atr(high, low, close, length):
        tr = wilder_tr(close, high, low)
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr
    
    def normalized_volume_energy(vol, length):
        high_vol = vol.rolling(length).max()
        low_vol = vol.rolling(length).min()
        range_vol = high_vol - low_vol
        range_vol = range_vol.replace(0, 1)
        nve = ((vol - low_vol) / range_vol) * 100
        return nve
    
    # ========== CALCULATE INDICATORS ==========
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # ALMA Baseline
    alma_val = alma(close, alma_window, alma_offset, alma_sigma)
    
    # ALMA Histogram (momentum)
    alma_hist = close - alma_val
    
    # FluxWave Oscillator
    fluxwave_base = alma(close, flux_length, flux_offset, float(flux_sigma))
    fluxwave_smooth = fluxwave_base.ewm(span=flux_smooth, adjust=False).mean()
    
    # FluxWave bands
    fluxwave_std = fluxwave_base.rolling(flux_length).std()
    fluxwave_upper = fluxwave_smooth + fluxwave_std * flux_band_mult
    fluxwave_lower = fluxwave_smooth - fluxwave_std * flux_band_mult
    
    # FluxWave RSI-like signal
    fluxwave_rsi = wilder_rsi(fluxwave_base - fluxwave_smooth, flux_length)
    
    # QuickSilver Bands
    qs_ma = close.rolling(qs_period).mean()
    qs_std = close.rolling(qs_period).std()
    qs_upper = qs_ma + qs_std * qs_deviation
    qs_lower = qs_ma - qs_std * qs_deviation
    qs_width = qs_upper - qs_lower
    
    # Volume Energy
    vol_energy = normalized_volume_energy(volume, vol_length)
    
    # ========== ENTRY CONDITIONS ==========
    # ALMA Crossover / Crossunder
    alma_crossover = (close > alma_val) & (close.shift(1) <= alma_val.shift(1))
    alma_crossunder = (close < alma_val) & (close.shift(1) >= alma_val.shift(1))
    
    # FluxWave OS/OB signals
    fluxwave_ob = fluxwave_rsi > 70
    fluxwave_os = fluxwave_rsi < 30
    
    # QuickSilver breakout
    qs_breakout_up = close > qs_upper
    qs_breakout_down = close < qs_lower
    
    # Volume filter
    vol_confirm = vol_energy > vol_normal
    
    # Build composite signals
    long_cond = pd.Series(False, index=df.index)
    short_cond = pd.Series(False, index=df.index)
    
    for i in range(flux_length, len(df)):
        if pd.isna(alma_val.iloc[i]) or pd.isna(fluxwave_rsi.iloc[i]):
            continue
        
        # ALMA confirmation
        alma_confirm_long = alma_crossover.iloc[i] if use_alma_signal else True
        alma_confirm_short = alma_crossunder.iloc[i] if use_alma_signal else True
        
        # FluxWave confirmation
        flux_confirm_long = fluxwave_os.iloc[i] if use_flux_signal else True
        flux_confirm_short = fluxwave_ob.iloc[i] if use_flux_signal else True
        
        # QuickSilver confirmation
        qs_confirm_long = qs_breakout_up.iloc[i] if use_qs_signal else True
        qs_confirm_short = qs_breakout_down.iloc[i] if use_qs_signal else True
        
        # Volume confirmation
        vol_ok = vol_confirm.iloc[i] if use_vol_filter else True
        
        # Combined conditions
        long_cond.iloc[i] = alma_confirm_long and flux_confirm_long and qs_confirm_long and vol_ok
        short_cond.iloc[i] = alma_confirm_short and flux_confirm_short and qs_confirm_short and vol_ok
    
    # ========== GENERATE ENTRIES ==========
    entries = []
    trade_num = 1
    last_bar_long = -qfn_cooldown
    last_bar_short = -qfn_cooldown
    
    for i in range(flux_length, len(df)):
        if pd.isna(alma_val.iloc[i]) or pd.isna(fluxwave_rsi.iloc[i]):
            continue
        
        ts = int(df['time'].iloc[i])
        entry_price = float(df['close'].iloc[i])
        
        # Long entry
        if long_cond.iloc[i] and (i - last_bar_long) > qfn_cooldown:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            last_bar_long = i
        
        # Short entry
        elif short_cond.iloc[i] and (i - last_bar_short) > qfn_cooldown:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
            last_bar_short = i
    
    return entries