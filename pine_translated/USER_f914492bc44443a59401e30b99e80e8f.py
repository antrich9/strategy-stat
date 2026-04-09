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
    # Default parameters from Pine Script inputs
    alma_window = 9
    alma_offset = 0.85
    alma_sigma = 6.0
    flux_smooth = 1
    flux_length = 50
    flux_offset = 0.85
    flux_sigma = 6
    flux_band_mult = 1.0
    qs_period = 20
    qs_deviation = 2.0
    vol_length = 50
    entry_lookback = 5
    use_alma_signal = True
    use_flux_signal = True
    use_qs_signal = True
    use_vol_filter = True
    entry_mode = "Combined"
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # ALMA Implementation
    def alma(src, win, off, sig):
        m = (win - 1) * off
        s = win / sig
        w = np.exp(-np.square(np.arange(win) - m) / (2 * s * s))
        w = w / w.sum()
        pad_len = win - 1
        padded = np.concatenate([np.full(pad_len, src.iloc[0]), src.values])
        return pd.Series(np.convolve(padded, w, mode='valid'), index=src.index)
    
    # RSI Implementation (Wilder)
    def wilder_rsi(src, len):
        delta = src.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        ema_gain = gain.ewm(alpha=1.0/len, adjust=False).mean()
        ema_loss = loss.ewm(alpha=1.0/len, adjust=False).mean()
        rs = ema_gain / ema_loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    # ATR Implementation (Wilder)
    def wilder_atr(high, low, close, len):
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0/len, adjust=False).mean()
        return atr
    
    # Volume Energy Calculation
    def volume_energy(vol, length):
        vol_ma = vol.rolling(length).mean()
        return (vol - vol_ma) / vol_ma * 100
    
    close_series = pd.Series(close.values, index=df.index)
    high_series = pd.Series(high.values, index=df.index)
    low_series = pd.Series(low.values, index=df.index)
    volume_series = pd.Series(volume.values, index=df.index)
    
    # Calculate ALMA Baseline
    alma_val = alma(close_series, alma_window, alma_offset, alma_sigma)
    
    # ALMA Trend
    alma_trend_up = close_series > alma_val
    alma_trend_down = close_series < alma_val
    alma_cross_up = alma_trend_up & ~alma_trend_up.shift(1).fillna(False).astype(bool)
    alma_cross_up = alma_cross_up.shift(1).fillna(False) & alma_trend_up
    alma_cross_down = alma_trend_down & ~alma_trend_down.shift(1).fillna(False).astype(bool)
    alma_cross_down = alma_cross_down.shift(1).fillna(False) & alma_trend_down
    
    alma_signal = pd.Series(False, index=df.index)
    alma_signal[alma_cross_up] = True
    alma_signal[alma_cross_down] = True
    
    # FluxWave Oscillator
    src = close_series
    flux_src = src.rolling(flux_smooth).mean()
    flux_alma = alma(flux_src, flux_length, flux_offset, float(flux_sigma))
    flux_rsi = wilder_rsi(flux_src - flux_alma, flux_length)
    flux_upper = flux_alma + flux_band_mult * flux_alma.std()
    flux_lower = flux_alma - flux_band_mult * flux_alma.std()
    flux_ob = flux_rsi > 70
    flux_os = flux_rsi < 30
    flux_cross_up = (flux_rsi > flux_rsi.shift(1)) & (flux_rsi.shift(1) <= flux_rsi.shift(2))
    flux_cross_down = (flux_rsi < flux_rsi.shift(1)) & (flux_rsi.shift(1) >= flux_rsi.shift(2))
    flux_long = flux_cross_up & flux_os
    flux_short = flux_cross_down & flux_ob
    
    # QuickSilver Bands
    qs_ma = close_series.rolling(qs_period).mean()
    qs_std = close_series.rolling(qs_period).std()
    qs_upper = qs_ma + qs_deviation * qs_std
    qs_lower = qs_ma - qs_deviation * qs_std
    qs_breakout_up = close_series > qs_upper
    qs_breakout_down = close_series < qs_lower
    
    # Volume Energy Filter
    vol_energy = volume_energy(volume_series, vol_length)
    vol_confirm = vol_energy > 0
    
    # Entry Confirmation (lookback bars)
    def confirm_bars(condition, lookback):
        confirmed = pd.Series(False, index=df.index)
        for i in range(lookback, len(condition)):
            if condition.iloc[i-lookback:i].all():
                confirmed.iloc[i] = True
        return confirmed
    
    # Build entry conditions based on mode
    long_conditions = []
    short_conditions = []
    
    if use_alma_signal:
        long_conditions.append(alma_signal & (close_series > alma_val))
        short_conditions.append(alma_signal & (close_series < alma_val))
    
    if use_flux_signal:
        long_conditions.append(flux_long)
        short_conditions.append(flux_short)
    
    if use_qs_signal:
        long_conditions.append(qs_breakout_up)
        short_conditions.append(qs_breakout_down)
    
    # Combine conditions
    long_entry = pd.Series(False, index=df.index)
    short_entry = pd.Series(False, index=df.index)
    
    for cond in long_conditions:
        long_entry = long_entry | cond
    for cond in short_conditions:
        short_entry = short_entry | cond
    
    # Apply volume filter
    if use_vol_filter:
        long_entry = long_entry & vol_confirm
        short_entry = short_entry & vol_confirm
    
    # Apply confirmation lookback
    long_entry = confirm_bars(long_entry, entry_lookback)
    short_entry = confirm_bars(short_entry, entry_lookback)
    
    # Generate entry list
    entries = []
    trade_num = 1
    last_long_bar = -999
    last_short_bar = -999
    cooldown = 3
    
    for i in range(max(entry_lookback, qs_period, flux_length) + 1, len(df)):
        if pd.isna(alma_val.iloc[i]) or pd.isna(vol_energy.iloc[i]):
            continue
        
        direction = None
        if long_entry.iloc[i] and (i - last_long_bar) > cooldown:
            direction = 'long'
            last_long_bar = i
        elif short_entry.iloc[i] and (i - last_short_bar) > cooldown:
            direction = 'short'
            last_short_bar = i
        
        if direction:
            ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
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