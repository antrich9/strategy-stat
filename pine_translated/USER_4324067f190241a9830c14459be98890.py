import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    bb = 20
    input_retSince = 2
    input_retValid = 2
    trade_direction = 'Both'

    entries = []
    trade_num = 0

    sBreak = False
    rBreak = False
    sBreak_bar = None
    rBreak_bar = None
    sTop_at_breakout = None
    sBot_at_breakout = None
    rTop_at_breakout = None
    rBot_at_breakout = None
    sBars_since_breakout = 0
    rBars_since_breakout = 0
    sRetOccurred = False
    rRetOccurred = False
    sRetValid = False
    rRetValid = False
    prev_pl = np.nan
    prev_ph = np.nan

    for i in range(len(df)):
        close = df['close'].iloc[i]
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]

        pl = df['pl'].iloc[i] if pd.notna(df['pl'].iloc[i]) else np.nan
        ph = df['ph'].iloc[i] if pd.notna(df['ph'].iloc[i]) else np.nan

        pl_changed = pd.notna(pl) and pd.notna(prev_pl) and pl != prev_pl
        ph_changed = pd.notna(ph) and pd.notna(prev_ph) and ph != prev_ph

        if pl_changed:
            sBreak = False
            sBreak_bar = None

        if ph_changed:
            rBreak = False
            rBreak_bar = None

        if pl_changed:
            sTop = low[bb - 1] if pd.notna(df['low'].iloc[i - bb - 1]) else low
            sBot = pl

        if ph_changed:
            rBot = high[bb - 1] if pd.notna(df['high'].iloc[i - bb - 1]) else high
            rTop = ph

        cu = close < sBot if sBot is not None and pd.notna(sBot) else False
        co = close > rTop if rTop is not None and pd.notna(rTop) else False

        if cu and not sBreak:
            sBreak = True
            sBreak_bar = i
            sTop_at_breakout = sTop
            sBot_at_breakout = sBot

        if co and not rBreak:
            rBreak = True
            rBreak_bar = i
            rTop_at_breakout = rTop
            rBot_at_breakout = rBot

        if pl_changed:
            if not sBreak:
                pass
            sRetOccurred = False
            sRetValid = False

        if ph_changed:
            if not rBreak:
                pass
            rRetOccurred = False
            rRetValid = False

        if sBreak and sBreak_bar is not None:
            sBars_since_breakout = i - sBreak_bar
        else:
            sBars_since_breakout = 0

        if rBreak and rBreak_bar is not None:
            rBars_since_breakout = i - rBreak_bar
        else:
            rBars_since_breakout = 0

        if sBreak and sTop_at_breakout is not None and sBars_since_breakout >= input_retSince:
            s1 = high >= sTop_at_breakout and close < sBot_at_breakout
            s2 = high >= sTop_at_breakout and close >= sBot_at_breakout and close < sTop_at_breakout
            s3 = high >= sBot_at_breakout and high <= sTop_at_breakout
            s4 = high >= sBot_at_breakout and high <= sTop_at_breakout and close < sBot_at_breakout

            sRetActive = s1 or s2 or s3 or s4

            if sRetActive and not sRetOccurred:
                sRetValid = True

            if sRetValid and sBars_since_breakout > input_retValid:
                sRetValid = False

            if sRetValid and not sRetOccurred:
                trade_num += 1
                entry_ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'long',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': close,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close,
                    'raw_price_b': close
                })
                sRetOccurred = True

        if rBreak and rTop_at_breakout is not None and rBars_since_breakout >= input_retSince:
            r1 = low <= rBot_at_breakout and close >= rTop_at_breakout
            r2 = low <= rBot_at_breakout and close <= rTop_at_breakout and close >= rBot_at_breakout
            r3 = low <= rTop_at_breakout and low >= rBot_at_breakout
            r4 = low <= rTop_at_breakout and low >= rBot_at_breakout and close > rTop_at_breakout

            rRetActive = r1 or r2 or r3 or r4

            if rRetActive and not rRetOccurred:
                rRetValid = True

            if rRetValid and rBars_since_breakout > input_retValid:
                rRetValid = False

            if rRetValid and not rRetOccurred:
                trade_num += 1
                entry_ts = int(df['time'].iloc[i])
                entry_time = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': 'short',
                    'entry_ts': entry_ts,
                    'entry_time': entry_time,
                    'entry_price_guess': close,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': close,
                    'raw_price_b': close
                })
                rRetOccurred = True

        prev_pl = pl
        prev_ph = ph

    return entries