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
    # Default toggles from inputs (treat as enabled for full logic)
    useObOnly = False
    useObFvgStacked = True
    useFvgOnly = False
    useInverseFvg = False
    useSharpTurn = False
    useDoubleFvg = False
    useBfvg = False
    useEngulfing = False
    sweepMode = "Both"

    # Convert time to datetime for hour extraction
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['hour'] = df['datetime'].dt.hour

    # NY Hour (America/New_York) - approximate by using UTC offset
    # During standard time NY is UTC-5, during DST UTC-4
    # Simplified: assume always UTC-5 for hour calculation
    df['ny_hour'] = (df['hour'] - 5) % 24

    # Asian Session Detection: NY hour >= 19 (19:00-00:00 NY time)
    df['in_asian_session'] = df['ny_hour'] >= 19

    # Session state tracking
    asian_session_started = df['in_asian_session'] & (~df['in_asian_session'].shift(1).fillna(True).astype(bool))
    asian_session_ended = (~df['in_asian_session']) & df['in_asian_session'].shift(1).fillna(False).astype(bool)

    # Asian High/Low tracking
    df['asian_high'] = np.nan
    df['asian_low'] = np.nan
    df['temp_high'] = np.nan
    df['temp_low'] = np.nan

    asian_high_val = np.nan
    asian_low_val = np.nan
    temp_high_val = np.nan
    temp_low_val = np.nan

    for i in range(len(df)):
        if asian_session_started.iloc[i]:
            temp_high_val = df['high'].iloc[i]
            temp_low_val = df['low'].iloc[i]
        elif df['in_asian_session'].iloc[i]:
            if not np.isnan(temp_high_val):
                temp_high_val = max(temp_high_val, df['high'].iloc[i])
            if not np.isnan(temp_low_val):
                temp_low_val = min(temp_low_val, df['low'].iloc[i])
        
        if asian_session_ended.iloc[i]:
            asian_high_val = temp_high_val
            asian_low_val = temp_low_val
        
        df.at[df.index[i], 'temp_high'] = temp_high_val
        df.at[df.index[i], 'temp_low'] = temp_low_val
        df.at[df.index[i], 'asian_high'] = asian_high_val
        df.at[df.index[i], 'asian_low'] = asian_low_val

    # Sweep detection for Asian High/Low
    df['swept_high'] = False
    df['swept_low'] = False
    df['both_swept'] = False

    swept_high = False
    swept_low = False
    both_swept = False
    prev_asian_high = np.nan
    prev_asian_low = np.nan

    for i in range(1, len(df)):
        if asian_session_ended.iloc[i]:
            swept_high = False
            swept_low = False
            both_swept = False

        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]

        if not swept_high and not np.isnan(prev_asian_high) and current_high > prev_asian_high:
            swept_high = True
        if not swept_low and not np.isnan(prev_asian_low) and current_low < prev_asian_low:
            swept_low = True
        if swept_high and swept_low and not both_swept:
            both_swept = True

        df.at[df.index[i], 'swept_high'] = swept_high
        df.at[df.index[i], 'swept_low'] = swept_low
        df.at[df.index[i], 'both_swept'] = both_swept

        prev_asian_high = df['asian_high'].iloc[i]
        prev_asian_low = df['asian_low'].iloc[i]

    # Previous Day High/Low
    df['new_day'] = df['datetime'].dt.date != df['datetime'].dt.date.shift(1)
    df['new_day'].fillna(True, inplace=True)

    df['pd_high'] = np.nan
    df['pd_low'] = np.nan
    df['temp_high1'] = np.nan
    df['temp_low1'] = np.nan

    pd_high_val = np.nan
    pd_low_val = np.nan
    temp_high1_val = np.nan
    temp_low1_val = np.nan

    for i in range(len(df)):
        if df['new_day'].iloc[i]:
            pd_high_val = temp_high1_val
            pd_low_val = temp_low1_val
            temp_high1_val = df['high'].iloc[i]
            temp_low1_val = df['low'].iloc[i]
        else:
            if np.isnan(temp_high1_val):
                temp_high1_val = df['high'].iloc[i]
            else:
                temp_high1_val = max(temp_high1_val, df['high'].iloc[i])
            if np.isnan(temp_low1_val):
                temp_low1_val = df['low'].iloc[i]
            else:
                temp_low1_val = min(temp_low1_val, df['low'].iloc[i])

        df.at[df.index[i], 'pd_high'] = pd_high_val
        df.at[df.index[i], 'pd_low'] = pd_low_val
        df.at[df.index[i], 'temp_high1'] = temp_high1_val
        df.at[df.index[i], 'temp_low1'] = temp_low1_val

    # PD Sweep detection
    df['pd_swept_high'] = False
    df['pd_swept_low'] = False

    pd_swept_high = False
    pd_swept_low = False
    prev_pd_high = np.nan
    prev_pd_low = np.nan

    for i in range(1, len(df)):
        if df['new_day'].iloc[i]:
            pd_swept_high = False
            pd_swept_low = False

        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]

        if not pd_swept_high and not np.isnan(prev_pd_high) and current_high > prev_pd_high:
            pd_swept_high = True
        if not pd_swept_low and not np.isnan(prev_pd_low) and current_low < prev_pd_low:
            pd_swept_low = True

        df.at[df.index[i], 'pd_swept_high'] = pd_swept_high
        df.at[df.index[i], 'pd_swept_low'] = pd_swept_low

        prev_pd_high = df['pd_high'].iloc[i]
        prev_pd_low = df['pd_low'].iloc[i]

    # London time (Europe/London) - approximate by assuming UTC+0
    df['london_hour'] = df['hour']

    # Trading window: 07:00-10:00 London time
    london_start_morning = 7
    london_end_morning = 10

    df['in_trading_window'] = (df['london_hour'] >= london_start_morning) & (df['london_hour'] < london_end_morning)

    # ATR (Wilder ATR)
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift(1))
    tr3 = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/14, adjust=False).mean()

    # Pattern Definitions
    # Candle direction helpers
    is_up = df['close'] > df['open']
    is_down = df['close'] < df['open']

    # OB (Order Block)
    ob_up = is_down.shift(2) & is_up.shift(1) & (df['close'].shift(1) > df['high'].shift(2))
    ob_down = is_up.shift(2) & is_down.shift(1) & (df['close'].shift(1) < df['low'].shift(2))

    # Standard FVG
    fvg_up = df['low'] > df['high'].shift(2)
    fvg_down = df['high'] < df['low'].shift(2)

    # Stacked OB + FVG
    stacked_up = ob_up & fvg_up
    stacked_down = ob_down & fvg_down

    # Inverse FVG
    inverse_fvg_up = (df['high'].shift(2) < df['low']) & (df['close'] >= df['high'].shift(2))
    inverse_fvg_down = (df['low'].shift(2) > df['high']) & (df['close'] <= df['low'].shift(2))

    # Sharp Turn (consecutive FVGs in opposite directions)
    sharp_turn_up = fvg_up.shift(1) & fvg_down
    sharp_turn_down = fvg_down.shift(1) & fvg_up

    # Double FVG (two consecutive FVGs in same direction)
    double_fvg_up = fvg_up & fvg_up.shift(1)
    double_fvg_down = fvg_down & fvg_down.shift(1)

    # Balanced FVG (gap fully contained within prior candle range)
    bfvg_up = (df['low'] > df['low'].shift(2)) & (df['low'] < df['high'].shift(2)) & (df['high'] > df['low'].shift(2)) & (df['high'] < df['high'].shift(2))
    bfvg_down = (df['high'] < df['high'].shift(2)) & (df['high'] > df['low'].shift(2)) & (df['low'] < df['high'].shift(2)) & (df['low'] > df['low'].shift(2))

    # Engulfing candle
    engulfing_up = is_up & is_down.shift(1) & (df['close'] > df['high'].shift(1)) & (df['open'] < df['low'].shift(1))
    engulfing_down = is_down & is_up.shift(1) & (df['close'] < df['low'].shift(1)) & (df['open'] > df['high'].shift(1))

    # Build combined pattern conditions
    bullish_pattern = pd.Series(False, index=df.index)
    bearish_pattern = pd.Series(False, index=df.index)

    if useObOnly:
        bullish_pattern |= ob_up
        bearish_pattern |= ob_down
    if useObFvgStacked:
        bullish_pattern |= stacked_up
        bearish_pattern |= stacked_down
    if useFvgOnly:
        bullish_pattern |= fvg_up
        bearish_pattern |= fvg_down
    if useInverseFvg:
        bullish_pattern |= inverse_fvg_up
        bearish_pattern |= inverse_fvg_down
    if useSharpTurn:
        bullish_pattern |= sharp_turn_up
        bearish_pattern |= sharp_turn_down
    if useDoubleFvg:
        bullish_pattern |= double_fvg_up
        bearish_pattern |= double_fvg_down
    if useBfvg:
        bullish_pattern |= bfvg_up
        bearish_pattern |= bfvg_down
    if useEngulfing:
        bullish_pattern |= engulfing_up
        bearish_pattern |= engulfing_down

    # Sweep filter conditions
    asian_swept_high = df['swept_high']
    asian_swept_low = df['swept_low']
    pd_swept_high_cond = df['pd_swept_high']
    pd_swept_low_cond = df['pd_swept_low']

    # Determine entry conditions based on sweep mode
    if sweepMode == "Asian Only":
        long_sweep_cond = asian_swept_high
        short_sweep_cond = asian_swept_low
    elif sweepMode == "PD Only":
        long_sweep_cond = pd_swept_high_cond
        short_sweep_cond = pd_swept_low_cond
    else:  # "Both"
        long_sweep_cond = asian_swept_high & pd_swept_high_cond
        short_sweep_cond = asian_swept_low & pd_swept_low_cond

    # Entry signals
    long_entry = bullish_pattern & long_sweep_cond & df['in_trading_window'] & (~df['both_swept'])
    short_entry = bearish_pattern & short_sweep_cond & df['in_trading_window'] & (~df['both_swept'])

    # Generate entry list
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if long_entry.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': entry_time,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })
            trade_num += 1
        elif short_entry.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
            entry_price = float(df['close'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
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