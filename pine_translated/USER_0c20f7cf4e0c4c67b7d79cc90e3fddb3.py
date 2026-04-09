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
    close = df['close'].copy()
    high = df['high'].copy()
    low = df['low'].copy()
    time_col = df['time'].copy()

    # Date range parameters (default from Pine Script)
    start_year, start_month, start_date = 2022, 1, 1
    end_year, end_month, end_date = 2030, 1, 1
    start_ts = int(datetime(start_year, start_month, start_date, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime(end_year, end_month, end_date, tzinfo=timezone.utc).timestamp())
    in_date_range = (time_col >= start_ts) & (time_col < end_ts)

    # QQE MOD Settings
    RSI_Period = 6
    SF = 6
    QQE = 3
    Wilders_Period = RSI_Period * 2 - 1  # 11

    # Wilder RSI implementation
    def wilders_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    Rsi = wilders_rsi(close, RSI_Period)
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    AtrRsi = np.abs(RsiMa.shift(1) - RsiMa)
    MaAtrRsi = AtrRsi.ewm(alpha=1.0/Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi.ewm(alpha=1.0/Wilders_Period, adjust=False).mean() * QQE

    # SSL Settings
    SSL_Look_Back = 10
    SSL_Factor = 3.0
    SSLEMA = close.rolling(SSL_Look_Back).mean()
    SSL = SSLEMA - SSL_Factor * close.rolling(SSL_Look_Back).std()

    # WAE Settings
    BBLength = 14
    BBMult = 2.0
    RSILength = 14
    RSIOversold = 30
    Src = close
    basis = Src.rolling(BBLength).mean()
    dev = BBMult * Src.rolling(BBLength).std()
    upper = basis + dev
    lower = basis - dev
    rsi_wae = wilders_rsi(Src, RSILength)

    # Build boolean conditions
    crossover_ssl_sslema = (SSL > SSLEMA) & (SSL.shift(1) <= SSLEMA.shift(1))
    crossover_upper_lower = (upper > upper.shift(1)) & (lower > lower.shift(1))
    crossover_rsi_oversold = (rsi_wae > RSIOversold) & (rsi_wae.shift(1) <= RSIOversold)
    waexplosion = crossover_upper_lower & crossover_rsi_oversold

    # QQE longband and trend (stateful)
    longband_arr = np.zeros(len(df))
    trend_arr = np.zeros(len(df))
    longCondition = pd.Series(False, index=df.index)

    for i in range(len(df)):
        if i < Wilders_Period:
            continue
        RSIndex_val = RsiMa.iloc[i]
        DeltaFastAtrRsi_val = dar.iloc[i]
        newlongband = RSIndex_val - DeltaFastAtrRsi_val
        if i == 0:
            longband_prev = 0.0
            trend_prev = 0
        else:
            longband_prev = longband_arr[i-1]
            trend_prev = trend_arr[i-1]
        if RSIndex_val > longband_prev and RSIndex_val > longband_prev:
            longband_arr[i] = max(longband_prev, newlongband)
        else:
            longband_arr[i] = newlongband
        trend_arr[i] = 1 if RSIndex_val > longband_arr[i] else trend_prev
        longCondition.iloc[i] = (trend_arr[i] == 1)

    # Combined long entry condition
    long_entry_cond = crossover_ssl_sslema & waexplosion & longCondition & in_date_range

    # Generate entries
    entries = []
    trade_num = 1
    for i in range(len(df)):
        if long_entry_cond.iloc[i]:
            entry_ts = int(time_col.iloc[i])
            entry_price = float(close.iloc[i])
            entry_dict = {
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            }
            entries.append(entry_dict)
            trade_num += 1
    return entries