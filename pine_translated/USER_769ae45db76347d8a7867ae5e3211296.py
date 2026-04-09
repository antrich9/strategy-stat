import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    """
    df columns: time(int unix ts), open(float), high(float), low(float), close(float), volume(float)
    Rows sorted ascending by time (oldest first). Index is 0-based int.
    """
    
    # Input parameters (matching Pine Script defaults)
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
    vol_normal = 75
    
    entry_lookback = 5
    entry_threshold = 0.5
    
    use_alma_signal = True
    use_flux_signal = True
    use_qs_signal = True
    use_vol_filter = True
    
    # =====================================
    # INDICATOR CALCULATIONS
    # =====================================
    
    # ----- ALMA Calculation -----
    def alma_filter(series, window, offset, sigma):
        m = (offset * (window - 1))
        s = sigma * (window - 1) / 6
        weights = np.exp(-(np.arange(window) - m)**2 / (2 * s**2))
        weights = weights / weights.sum()
        
        result = np.zeros(len(series))
        for i in range(window - 1, len(series)):
            result[i] = np.sum(weights * series[max(0, i - window + 1):i + 1])
        return pd.Series(result, index=series.index)
    
    alma_line = alma_filter(df['close'].values, alma_window, alma_offset, alma_sigma)
    alma_line = pd.Series(alma_line, index=df.index)
    
    # ----- FluxWave Oscillator Calculation -----
    flux_alma = alma_filter(df['close'].values, flux_length, flux_offset, flux_sigma)
    flux_alma = pd.Series(flux_alma, index=df.index)
    
    flux_base = df['close'] - flux_alma
    
    # Apply smoothing
    for _ in range(flux_smooth):
        flux_base = flux_base.ewm(span=2, adjust=False).mean()
    
    flux_atr = df['high'] - df['low']
    flux_atr_avg = flux_atr.ewm(span=flux_length, adjust=False).mean()
    
    flux_wave = flux_base / (flux_atr_avg + 1e-10)
    flux_wave = flux_wave.ewm(span=flux_length, adjust=False).mean()
    
    # ----- QuickSilver Bands Calculation -----
    qs_sma = df['close'].rolling(qs_period).mean()
    qs_std = df['close'].rolling(qs_period).std()
    
    qs_upper = qs_sma + qs_deviation * qs_std
    qs_middle = qs_sma
    qs_lower = qs_sma - qs_deviation * qs_std
    
    # ----- Volume Energy Calculation -----
    vol_ema = df['volume'].ewm(span=vol_length, adjust=False).mean()
    vol_ratio = df['volume'] / (vol_ema + 1e-10)
    vol_energy = (vol_ratio / vol_ratio.rolling(vol_length).max()) * 100
    
    # ----- Entry Condition Series -----
    price_above_alma = df['close'] > alma_line
    price_below_alma = df['close'] < alma_line
    
    flux_oversold = flux_wave < -0.5
    flux_overbought = flux_wave > 0.5
    
    qs_breakout_up = df['close'] > qs_upper
    qs_breakout_down = df['close'] < qs_lower
    
    vol_confirm = vol_energy > vol_normal
    
    # Combined conditions
    long_cond = (
        (use_alma_signal and price_above_alma) &
        (use_flux_signal and flux_oversold) &
        (use_qs_signal and qs_breakout_up) &
        (use_vol_filter and vol_confirm)
    )
    
    short_cond = (
        (use_alma_signal and price_below_alma) &
        (use_flux_signal and flux_overbought) &
        (use_qs_signal and qs_breakout_down) &
        (use_vol_filter and vol_confirm)
    )
    
    # Crossover detection
    long_entry = pd.Series(False, index=df.index)
    short_entry = pd.Series(False, index=df.index)
    
    for i in range(1, len(df)):
        if pd.isna(alma_line.iloc[i]) or pd.isna(flux_wave.iloc[i]):
            continue
        
        if (df['close'].iloc[i] > alma_line.iloc[i] and 
            df['close'].iloc[i-1] <= alma_line.iloc[i-1]):
            long_entry.iloc[i] = True
        
        if (df['close'].iloc[i] < alma_line.iloc[i] and 
            df['close'].iloc[i-1] >= alma_line.iloc[i-1]):
            short_entry.iloc[i] = True
    
    # Final long/short signals
    long_signal = long_entry & long_cond
    short_signal = short_entry & short_cond
    
    # =====================================
    # GENERATE ENTRIES
    # =====================================
    
    entries = []
    trade_num = 1
    last_entry_idx = -100  # Cooldown buffer
    
    for i in range(len(df)):
        if pd.isna(alma_line.iloc[i]) or pd.isna(flux_wave.iloc[i]):
            continue
        
        if i - last_entry_idx < 3:
            continue
        
        direction = None
        if long_signal.iloc[i]:
            direction = 'long'
            last_entry_idx = i
        elif short_signal.iloc[i]:
            direction = 'short'
            last_entry_idx = i
        
        if direction:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = df['close'].iloc[i]
            
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
    
    return entries