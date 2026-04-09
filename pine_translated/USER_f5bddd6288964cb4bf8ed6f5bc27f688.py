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
    trend_period = 20
    ma_type = 'EMA'
    ma_period = 8
    trigger_up = 0.05
    trigger_down = -0.05
    
    # E2PSS parameters
    use_e2pss = True
    inverse_e2pss = False
    period_e2pss = 15
    
    close = df['close'].values
    n = len(close)
    
    # Calculate TDF
    mma = pd.Series(close).ewm(span=trend_period, adjust=False).mean().values
    smma = pd.Series(mma).ewm(span=trend_period, adjust=False).mean().values
    
    impet_mma = np.diff(mma, prepend=mma[0])
    impet_smma = np.diff(smma, prepend=smma[0])
    
    min_tick = 1e-10
    div_ma = np.abs(mma - smma) / min_tick
    aver_impet = ((impet_mma + impet_smma) / 2) / (2 * min_tick)
    
    tdf_raw = div_ma * np.power(aver_impet, 3)
    
    period = 3 * trend_period - 1
    tdf_abs_raw = np.zeros(n)
    for i in range(period, n):
        max_val = np.abs(tdf_raw[i])
        for j in range(1, period + 1):
            cand = np.abs(tdf_raw[i - j])
            if cand > max_val:
                max_val = cand
        tdf_abs_raw[i] = max_val
    
    tdf_abs_raw = pd.Series(tdf_abs_raw).ffill().bfill().values
    ratio = np.where(tdf_abs_raw != 0, tdf_raw / tdf_abs_raw, np.nan)
    ratio = np.nan_to_num(ratio, nan=0.0)
    
    if ma_type == 'EMA':
        ratio_series = pd.Series(ratio)
        smooth = ratio_series.ewm(span=ma_period, adjust=False).mean().values
    elif ma_type == 'DEMA':
        ema1 = pd.Series(ratio).ewm(span=ma_period, adjust=False).mean().values
        ema2 = pd.Series(ema1).ewm(span=ma_period, adjust=False).mean().values
        smooth = 2 * ema1 - ema2
    elif ma_type == 'TEMA':
        ema1 = pd.Series(ratio).ewm(span=ma_period, adjust=False).mean().values
        ema2 = pd.Series(ema1).ewm(span=ma_period, adjust=False).mean().values
        ema3 = pd.Series(ema2).ewm(span=ma_period, adjust=False).mean().values
        smooth = 3 * (ema1 - ema2) + ema3
    elif ma_type == 'SMA':
        smooth = pd.Series(ratio).rolling(ma_period).mean().values
    elif ma_type == 'VWMA':
        volume = df['volume'].values
        smooth = pd.Series(ratio * volume).rolling(ma_period).sum().values / pd.Series(volume).rolling(ma_period).sum().values
    else:
        smooth = pd.Series(ratio).ewm(span=ma_period, adjust=False).mean().values
    
    tdf = np.where(tdf_abs_raw > 0, np.clip(smooth, -1, 1), 0.0)
    
    # Calculate Trendilo for E2PSS filter
    trendilo_smooth = 1
    trendilo_length = 50
    trendilo_offset = 0.85
    trendilo_sigma = 6
    trendilo_bmult = 1.0
    
    pct_change = pd.Series(close).pct_change(trendilo_smooth) * 100
    avg_pct_change = pct_change.rolling(trendilo_length).apply(
        lambda x: np.sum(x * np.exp(-((np.arange(len(x)) - len(x) * (1 - trendilo_offset)) ** 2) / (2 * (trendilo_sigma ** 2)))) / np.sum(np.exp(-((np.arange(len(x)) - len(x) * (1 - trendilo_offset)) ** 2) / (2 * (trendilo_sigma ** 2)))), raw=True
    )
    rms = trendilo_bmult * np.sqrt((avg_pct_change ** 2).rolling(trendilo_length).mean())
    
    trendilo_dir = np.where(avg_pct_change > rms, 1, np.where(avg_pct_change < -rms, -1, 0))
    
    # Calculate zero crossovers
    tdf_series = pd.Series(tdf)
    zero_cross_up = (tdf_series > 0) & (tdf_series.shift(1) <= 0)
    zero_cross_down = (tdf_series < 0) & (tdf_series.shift(1) >= 0)
    
    # Calculate E2PSS filter
    filt2 = np.zeros(n)
    trigger_e2pss = np.zeros(n)
    pi = 2 * np.arcsin(1)
    a1 = np.exp(-1.414 * pi / period_e2pss)
    b1 = 2 * a1 * np.cos(1.414 * pi / period_e2pss)
    coef2 = b1
    coef3 = -a1 * a1
    coef1 = 1 - coef2 - coef3
    
    for i in range(n):
        if i < 2:
            filt2[i] = close[i]
        else:
            filt2[i] = coef1 * close[i] + coef2 * filt2[i-1] + coef3 * filt2[i-2]
        trigger_e2pss[i] = filt2[i-1] if i > 0 else 0
    
    filt2_series = pd.Series(filt2)
    trigger_e2pss_series = pd.Series(trigger_e2pss)
    
    signal_long_e2pss = filt2_series > trigger_e2pss_series
    signal_short_e2pss = filt2_series < trigger_e2pss_series
    signal_long_e2pss_final = signal_short_e2pss if inverse_e2pss else signal_long_e2pss
    signal_short_e2pss_final = signal_long_e2pss if inverse_e2pss else signal_short_e2pss
    
    # Entry conditions
    tdf_s = pd.Series(tdf)
    cross_up = (tdf_s > trigger_up) & (tdf_s.shift(1) <= trigger_up)
    cross_down = (tdf_s < trigger_down) & (tdf_s.shift(1) >= trigger_down)
    
    long_condition = cross_up
    short_condition = cross_down
    
    if use_e2pss:
        long_condition = long_condition & signal_long_e2pss_final
        short_condition = short_condition & signal_short_e2pss_final
    
    # Build entries
    tdf_filled = tdf_s.ffill().bfill()
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if long_condition.iloc[i] and not short_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': int(df['time'].iloc[i]),
                'entry_time': datetime.fromtimestamp(df['time'].iloc[i], tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries