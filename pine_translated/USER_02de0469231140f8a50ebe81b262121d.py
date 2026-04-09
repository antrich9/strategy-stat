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
    df = df.sort_values('time').reset_index(drop=True)

    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['date'] = df['datetime'].dt.date

    daily = df.groupby('date').agg(pdHigh=('high', 'max'), pdLow=('low', 'min')).reset_index()
    daily['prev_pdHigh'] = daily['pdHigh'].shift(1)
    daily['prev_pdLow'] = daily['pdLow'].shift(1)

    df = df.merge(daily[['date', 'prev_pdHigh', 'prev_pdLow']], on='date', how='left')
    df.rename(columns={'prev_pdHigh': 'pdHigh', 'prev_pdLow': 'pdLow'}, inplace=True)

    df['pdRange'] = df['pdHigh'] - df['pdLow']
    df['upper50'] = df['pdLow'] + df['pdRange'] * 0.5
    df['lower40'] = df['pdLow'] + df['pdRange'] * 0.4

    entries = []
    trade_num = 1

    for date, group in df.groupby('date'):
        group = group.sort_values('time')

        pdHigh_i = group['pdHigh'].iloc[0]
        pdLow_i = group['pdLow'].iloc[0]

        if pd.isna(pdHigh_i) or pd.isna(pdLow_i):
            continue

        upper50_i = group['upper50'].iloc[0]
        lower40_i = group['lower40'].iloc[0]

        trade_taken = False

        for idx, row in group.iterrows():
            if trade_taken:
                break

            close = row['close']

            if close > upper50_i:
                direction = 'long'
                entry_price = float(close)
                entry_ts = int(row['time'])
                entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': direction,
                    'entry_ts': entry_ts,
                    'entry_time': entry_time_str,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
                trade_taken = True
            elif close < lower40_i:
                direction = 'short'
                entry_price = float(close)
                entry_ts = int(row['time'])
                entry_time_str = datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat()
                entries.append({
                    'trade_num': trade_num,
                    'direction': direction,
                    'entry_ts': entry_ts,
                    'entry_time': entry_time_str,
                    'entry_price_guess': entry_price,
                    'exit_ts': 0,
                    'exit_time': '',
                    'exit_price_guess': 0.0,
                    'raw_price_a': entry_price,
                    'raw_price_b': entry_price
                })
                trade_num += 1
                trade_taken = True

    return entries