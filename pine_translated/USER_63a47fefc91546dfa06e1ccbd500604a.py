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
    htf_enabled = True
    liquidity_lookback = 20
    fvg_min_size = 0.05
    fib_retracement_level = 0.71
    
    # Wilder's RSI implementation placeholder (needed for ta.rsi if used)
    # ATR calculation (Wilder's)
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - prev_close)
    tr3 = abs(df['low'] - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # HTF EMA indicators (placeholder - not computable without HTF data)
    htf_trend_bull = True
    htf_trend_bear = True
    
    # EMA indicators for trend
    ema_50 = df['close'].ewm(span=50, adjust=False).mean()
    ema_200 = df['close'].ewm(span=200, adjust=False).mean()
    
    # Pivot high/low detection (rolling window approach)
    window = liquidity_lookback + 1
    df['swing_high'] = df['high'].rolling(window=window, min_periods=1).max().shift(-liquidity_lookback)
    df['swing_low'] = df['low'].rolling(window=window, min_periods=1).min().shift(-liquidity_lookback)
    
    # Track recent swing levels (var float in Pine)
    df['recent_swing_high'] = np.nan
    df['recent_swing_low'] = np.nan
    
    # Update recent swings when new pivots form
    df.loc[df['swing_high'].notna(), 'recent_swing_high'] = df.loc[df['swing_high'].notna(), 'swing_high']
    df.loc[df['swing_low'].notna(), 'recent_swing_low'] = df.loc[df['swing_low'].notna(), 'swing_low']
    
    # Forward fill to propagate recent swings (simulates var behavior)
    df['recent_swing_high'] = df['recent_swing_high'].ffill()
    df['recent_swing_low'] = df['recent_swing_low'].ffill()
    
    # Liquidity sweep detection
    bull_liq_cond = df['recent_swing_low'].notna()
    bear_liq_cond = df['recent_swing_high'].notna()
    df['liquidity_sweep_bull'] = bull_liq_cond & (df['low'] < df['recent_swing_low']) & (df['close'] > df['recent_swing_low'])
    df['liquidity_sweep_bear'] = bear_liq_cond & (df['high'] > df['recent_swing_high']) & (df['close'] < df['recent_swing_high'])
    
    # Break of Structure (BOS)
    df['bull_bos'] = df['liquidity_sweep_bull'] & (df['close'] > df['high'].shift(1))
    df['bear_bos'] = df['liquidity_sweep_bear'] & (df['close'] < df['low'].shift(1))
    
    # Fair Value Gap (FVG) detection
    df['bull_fvg'] = (df['low'] > df['high'].shift(2)) & ((df['low'] - df['high'].shift(2)) / df['close']) > fvg_min_size / 100
    df['bear_fvg'] = (df['high'] < df['low'].shift(2)) & ((df['low'].shift(2) - df['high']) / df['close']) > fvg_min_size / 100
    
    # Fibonacci 71% level (only calculated on BOS)
    df['fib_71_level'] = np.nan
    bull_fib_mask = df['bull_bos'] & df['recent_swing_high'].notna() & df['recent_swing_low'].notna()
    bear_fib_mask = df['bear_bos'] & df['recent_swing_high'].notna() & df['recent_swing_low'].notna()
    df.loc[bull_fib_mask, 'fib_71_level'] = df.loc[bull_fib_mask, 'recent_swing_high'] - (df.loc[bull_fib_mask, 'recent_swing_high'] - df.loc[bull_fib_mask, 'recent_swing_low']) * fib_retracement_level
    df.loc[bear_fib_mask, 'fib_71_level'] = df.loc[bear_fib_mask, 'recent_swing_low'] + (df.loc[bear_fib_mask, 'recent_swing_high'] - df.loc[bear_fib_mask, 'recent_swing_low']) * fib_retracement_level
    
    # HTF conditions
    bull_htf_ok = htf_enabled and htf_trend_bull
    bear_htf_ok = htf_enabled and htf_trend_bear
    
    # FVG present conditions (shifted to check prior bars)
    bull_fvg_present = df['bull_fvg'].shift(1) | df['bull_fvg'].shift(2) | df['bull_fvg'].shift(3)
    bear_fvg_present = df['bear_fvg'].shift(1) | df['bear_fvg'].shift(2) | df['bear_fvg'].shift(3)
    
    # Fibonacci level conditions
    bull_fib_ok = df['fib_71_level'].notna() & (df['close'] <= df['fib_71_level']) & (df['close'] >= df['fib_71_level'] * 0.99)
    bear_fib_ok = df['fib_71_level'].notna() & (df['close'] >= df['fib_71_level']) & (df['close'] <= df['fib_71_level'] * 1.01)
    
    # Combined entry conditions
    df['bullish_entry'] = bull_htf_ok & df['liquidity_sweep_bull'] & df['bull_bos'] & bull_fvg_present & bull_fib_ok
    df['bearish_entry'] = bear_htf_ok & df['liquidity_sweep_bear'] & df['bear_bos'] & bear_fvg_present & bear_fib_ok
    
    # Generate entries list
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if df['bullish_entry'].iloc[i]:
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
        elif df['bearish_entry'].iloc[i]:
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
    
    return entries