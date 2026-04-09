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
    close = df['close'].values
    volume = df['volume'].values
    n = len(df)
    
    # ALMA indicator
    alma_window = 9
    alma_offset = 0.85
    alma_sigma = 6.0
    
    m = (alma_window - 1) * alma_offset
    s = alma_sigma
    k = np.arange(alma_window)
    w = np.exp(-np.square(k - m) / (2 * s * s))
    w = w / w.sum()
    
    alma_vals = np.zeros(n)
    for i in range(alma_window - 1, n):
        alma_vals[i] = np.dot(close[i - alma_window + 1:i + 1], w)
    
    # FluxWave Oscillator
    flux_length = 50
    flux_smooth = 1
    flux_offset = 0.85
    flux_sigma = 6.0
    flux_band_mult = 1.0
    
    avg_price = pd.Series(close).ewm(span=flux_length, adjust=False).mean().values
    
    detrender = np.zeros(n)
    Q1 = np.zeros(n)
    I1 = np.zeros(n)
    
    for i in range(flux_length, n):
        detrender[i] = (close[i] + 3*close[i-1] + 3*close[i-2] + close[i-3]) / 32
        detrender[i] += 0.2 * detrender[i-1] + 0.1 * detrender[i-2]
        Q1[i] = detrender[i]
        I1[i] = detrender[i-1]
    
    osc = np.zeros(n)
    for i in range(flux_length + 1, n):
        Im = I1[i] * I1[i-1] + Q1[i] * Q1[i-1]
        Re = I1[i] * Q1[i-1] - I1[i-1] * Q1[i]
        j = np.sqrt(Re*Re + Im*Im)
        if j > 0:
            osc[i] = Q1[i] / j
        else:
            osc[i] = 0
    
    flux_vals = pd.Series(osc).ewm(span=flux_smooth, adjust=False).mean().values
    
    # QuickSilver Bands
    qs_period = 20
    qs_deviation = 2.0
    
    qs_middle_vals = pd.Series(close).rolling(qs_period).mean().values
    qs_std_vals = pd.Series(close).rolling(qs_period).std().values
    qs_upper_vals = qs_middle_vals + qs_deviation * qs_std_vals
    qs_lower_vals = qs_middle_vals - qs_deviation * qs_std_vals
    
    # Volume Energy
    vol_length = 50
    vol_normal = 75
    
    vol_mean = pd.Series(volume).rolling(vol_length).mean().values
    vol_std = pd.Series(volume).rolling(vol_length).std().values
    vol_energy_vals = np.zeros(n)
    for i in range(n):
        if vol_std[i] > 0:
            vol_energy_vals[i] = (volume[i] - vol_mean[i]) / vol_std[i] * 100
    
    # Entry parameters
    entry_lookback = 5
    entry_threshold = 0.5
    qfn_cooldown = 3
    
    entries = []
    trade_num = 1
    cooldown_remaining = 0
    
    for i in range(n):
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue
        
        # Skip bars with insufficient data
        if i < 50:
            continue
        
        # Check for NaN values
        if np.isnan(alma_vals[i]) or np.isnan(flux_vals[i]) or np.isnan(qs_middle_vals[i]) or np.isnan(vol_energy_vals[i]):
            continue
        
        price = close[i]
        ts = int(df['time'].iloc[i])
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc) if ts > 9999999999 else datetime.fromtimestamp(ts, tz=timezone.utc)
        
        # ALMA crossover conditions
        if i > 0:
            alma_cross_up = close[i] > alma_vals[i] and close[i-1] <= alma_vals[i-1]
            alma_cross_down = close[i] < alma_vals[i] and close[i-1] >= alma_vals[i-1]
        else:
            alma_cross_up = False
            alma_cross_down = False
        
        # FluxWave signal
        flux_signal = flux_vals[i] > 0
        
        # QuickSilver breakout
        qs_breakout_up = price > qs_middle_vals[i]
        qs_breakout_down = price < qs_middle_vals[i]
        
        # Volume filter
        vol_confirm = vol_energy_vals[i] > vol_normal
        
        # Build signal strength for lookback
        signal_strength = np.zeros(n)
        for j in range(n):
            strength = 0
            if not np.isnan(alma_vals[j]) and not np.isnan(flux_vals[j]) and not np.isnan(qs_middle_vals[j]) and not np.isnan(vol_energy_vals[j]):
                if close[j] > alma_vals[j]:
                    strength += 0.25
                if flux_vals[j] > 0:
                    strength += 0.25
                if close[j] > qs_middle_vals[j]:
                    strength += 0.25
                if vol_energy_vals[j] > vol_normal:
                    strength += 0.25
            signal_strength[j] = strength
        
        # Calculate average strength over lookback period
        if i >= entry_lookback:
            lookback_start = i - entry_lookback + 1
            lookback_strength = np.mean(signal_strength[lookback_start:i+1])
        else:
            lookback_strength = signal_strength[i]
        
        # Long entry: all conditions must be met
        long_conditions = (
            alma_cross_up and
            flux_signal and
            qs_breakout_up and
            vol_confirm and
            lookback_strength > entry_threshold
        )
        
        # Short entry: all conditions must be met
        short_conditions = (
            alma_cross_down and
            not flux_signal and
            qs_breakout_down and
            vol_confirm and
            lookback_strength > entry_threshold
        )
        
        if short_conditions:
            direction = 'short'
        elif long_conditions:
            direction = 'long'
        else:
            continue
        
        entries.append({
            'trade_num': trade_num,
            'direction': direction,
            'entry_ts': ts,
            'entry_time': dt.isoformat(),
            'entry_price_guess': price,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': price,
            'raw_price_b': price
        })
        trade_num += 1
        cooldown_remaining = qfn_cooldown
    
    return entries