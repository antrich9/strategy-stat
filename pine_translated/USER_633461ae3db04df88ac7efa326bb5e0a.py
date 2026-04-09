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
    # Parameters
    bb = 20
    input_retSince = 2
    input_retValid = 2
    input_repType = 'On'

    # Calculate pivot highs and lows
    pl_raw = pd.Series(np.nan, index=df.index)
    ph_raw = pd.Series(np.nan, index=df.index)

    for i in range(bb, len(df)):
        window_low = df['low'].iloc[i-bb:i+1]
        window_high = df['high'].iloc[i-bb:i+1]
        if df['low'].iloc[i-bb] == window_low.min():
            pl_raw.iloc[i-bb] = df['low'].iloc[i-bb]
        if df['high'].iloc[i-bb] == window_high.max():
            ph_raw.iloc[i-bb] = df['high'].iloc[i-bb]

    pl = pl_raw.ffill()
    ph = ph_raw.ffill()

    # Box Y locations
    s_yLoc = np.where(df['low'].shift(bb + 1) > df['low'].shift(bb - 1), df['low'].shift(bb - 1), df['low'].shift(bb + 1))
    r_yLoc = np.where(df['high'].shift(bb + 1) > df['high'].shift(bb - 1), df['high'].shift(bb + 1), df['high'].shift(bb - 1))

    # ATR (Wilder)
    tr = np.maximum(df['high'] - df['low'], np.maximum(np.abs(df['high'] - df['close'].shift(1)), np.abs(df['low'] - df['close'].shift(1))))
    atr = pd.Series(tr.ewm(alpha=1/14, adjust=False).mean())

    # Repaint mode
    rTon = input_repType == 'On'
    rTcc = input_repType == 'Off: Candle Confirmation'
    rThv = input_repType == 'Off: High & Low'

    def repaint_bar(c1, c2, c3):
        if rTon:
            return c1
        elif rThv:
            return c2
        elif rTcc:
            return c3
        else:
            return pd.Series(False, index=df.index)

    # Breakout conditions
    sBot = pl.ffill()
    rTop = ph.ffill()
    cu_raw = (df['close'] < sBot) & (df['close'].shift(1) >= sBot)
    co_raw = (df['close'] > rTop) & (df['close'].shift(1) <= rTop)
    cu = repaint_bar(cu_raw, (df['low'] < sBot) & (df['low'].shift(1) >= sBot), cu_raw & pd.Series([df.index[i] >= df.index[-1] if i == len(df)-1 else False for i in range(len(df))], index=df.index))
    co = repaint_bar(co_raw, (df['high'] > rTop) & (df['high'].shift(1) <= rTop), co_raw & pd.Series([df.index[i] >= df.index[-1] if i == len(df)-1 else False for i in range(len(df))], index=df.index))

    # Breakout flags (sBreak, rBreak)
    sBreak = pd.Series(False, index=df.index)
    rBreak = pd.Series(False, index=df.index)

    for i in range(1, len(df)):
        if cu.iloc[i] and pd.isna(sBreak.iloc[i-1]) or (isinstance(sBreak.iloc[i-1], (bool, np.bool_)) and not sBreak.iloc[i-1]):
            sBreak.iloc[i] = True
        else:
            sBreak.iloc[i] = sBreak.iloc[i-1] if not pd.isna(sBreak.iloc[i-1]) else False

        if co.iloc[i] and pd.isna(rBreak.iloc[i-1]) or (isinstance(rBreak.iloc[i-1], (bool, np.bool_)) and not rBreak.iloc[i-1]):
            rBreak.iloc[i] = True
        else:
            rBreak.iloc[i] = rBreak.iloc[i-1] if not pd.isna(rBreak.iloc[i-1]) else False

    for i in range(len(df)):
        if pd.notna(pl.iloc[i]) and pd.notna(pl.shift(1).iloc[i]):
            if pd.isna(sBreak.iloc[i]) or (isinstance(sBreak.iloc[i], (bool, np.bool_)) and not sBreak.iloc[i]):
                sBreak.iloc[i] = np.nan
            else:
                sBreak.iloc[i] = np.nan
        if pd.notna(ph.iloc[i]) and pd.notna(ph.shift(1).iloc[i]):
            if pd.isna(rBreak.iloc[i]) or (isinstance(rBreak.iloc[i], (bool, np.bool_)) and not rBreak.iloc[i]):
                rBreak.iloc[i] = np.nan
            else:
                rBreak.iloc[i] = np.nan

    # Retest conditions (support)
    s1 = (sBreak) & (df['high'] >= sTop) & (df['close'] <= sBot)
    s2 = (sBreak) & (df['high'] >= sTop) & (df['close'] >= sBot) & (df['close'] <= sTop)
    s3 = (sBreak) & (df['high'] >= sBot) & (df['high'] <= sTop)
    s4 = (sBreak) & (df['high'] >= sBot) & (df['high'] <= sTop) & (df['close'] < sBot)

    # Retest conditions (resistance)
    r1 = (rBreak) & (df['low'] <= rBot) & (df['close'] >= rTop)
    r2 = (rBreak) & (df['low'] <= rBot) & (df['close'] <= rTop) & (df['close'] >= rBot)
    r3 = (rBreak) & (df['low'] <= rTop) & (df['low'] >= rBot)
    r4 = (rBreak) & (df['low'] <= rTop) & (df['low'] >= rBot) & (df['close'] > rTop)

    sTop = pl.ffill()
    sBot = pl.ffill()
    rTop = ph.ffill()
    rBot = ph.ffill()

    s1 = (sBreak) & (df['high'] >= sTop) & (df['close'] <= sBot)
    s2 = (sBreak) & (df['high'] >= sTop) & (df['close'] >= sBot) & (df['close'] <= sTop)
    s3 = (sBreak) & (df['high'] >= sBot) & (df['high'] <= sTop)
    s4 = (sBreak) & (df['high'] >= sBot) & (df['high'] <= sTop) & (df['close'] < sBot)

    r1 = (rBreak) & (df['low'] <= rBot) & (df['close'] >= rTop)
    r2 = (rBreak) & (df['low'] <= rBot) & (df['close'] <= rTop) & (df['close'] >= rBot)
    r3 = (rBreak) & (df['low'] <= rTop) & (df['low'] >= rBot)
    r4 = (rBreak) & (df['low'] <= rTop) & (df['low'] >= rBot) & (df['close'] > rTop)

    # Helper function for retest event
    def process_retest_event(c1, c2, c3, c4, y1_series, y2_series, ptype):
        retOccurred = False
        retActive = c1 | c2 | c3 | c4
        retEvent = retActive & ~retActive.shift(1).fillna(False)
        retValue = pd.Series(np.nan, index=df.index)
        for i in range(len(df)):
            if retEvent.iloc[i]:
                retValue.iloc[i] = y1_series.iloc[i]

        retValue = retValue.ffill()

        retSince = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if retEvent.iloc[i]:
                retSince.iloc[i] = 0
            else:
                retSince.iloc[i] = retSince.iloc[i-1] + 1

        if ptype == 'ph':
            retConditions = repaint_bar(df['close'] >= retValue, df['high'] >= retValue, (df['close'] >= retValue) & pd.Series([df.index[i] >= df.index[-1] if i == len(df)-1 else False for i in range(len(df))], index=df.index))
        else:
            retConditions = repaint_bar(df['close'] <= retValue, df['low'] <= retValue, (df['close'] <= retValue) & pd.Series([df.index[i] >= df.index[-1] if i == len(df)-1 else False for i in range(len(df))], index=df.index))

        retValid = (retSince > 0) & (retSince <= input_retValid) & retConditions & (~retOccurred)
        return retValid, retEvent, retValue

    rRetValid, rRetEvent, rRetValue = process_retest_event(r1, r2, r3, r4, df['high'], df['low'], 'ph')
    sRetValid, sRetEvent, sRetValue = process_retest_event(s1, s2, s3, s4, df['low'], df['high'], 'pl')

    tradeDirection = 'Both'
    entries = []
    trade_num = 1

    for i in range(len(df)):
        if pd.isna(pl.iloc[i]) if i > 0 else True:
            continue

        entry = {
            'trade_num': 0,
            'direction': '',
            'entry_ts': 0,
            'entry_time': '',
            'entry_price_guess': 0.0,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': 0.0,
            'raw_price_b': 0.0
        }

        if rRetValid.iloc[i] and tradeDirection in ['Both', 'Long']:
            entry['trade_num'] = trade_num
            entry['direction'] = 'long'
            entry['entry_ts'] = int(df['time'].iloc[i])
            entry['entry_time'] = datetime.fromtimestamp(entry['entry_ts'], tz=timezone.utc).isoformat()
            entry['entry_price_guess'] = df['close'].iloc[i]
            entry['raw_price_a'] = df['close'].iloc[i]
            entry['raw_price_b'] = df['close'].iloc[i]
            entries.append(entry)
            trade_num += 1

        entry = {
            'trade_num': 0,
            'direction': '',
            'entry_ts': 0,
            'entry_time': '',
            'entry_price_guess': 0.0,
            'exit_ts': 0,
            'exit_time': '',
            'exit_price_guess': 0.0,
            'raw_price_a': 0.0,
            'raw_price_b': 0.0
        }

        if sRetValid.iloc[i] and tradeDirection in ['Both', 'Short']:
            entry['trade_num'] = trade_num
            entry['direction'] = 'short'
            entry['entry_ts'] = int(df['time'].iloc[i])
            entry['entry_time'] = datetime.fromtimestamp(entry['entry_ts'], tz=timezone.utc).isoformat()
            entry['entry_price_guess'] = df['close'].iloc[i]
            entry['raw_price_a'] = df['close'].iloc[i]
            entry['raw_price_b'] = df['close'].iloc[i]
            entries.append(entry)
            trade_num += 1

    return entries