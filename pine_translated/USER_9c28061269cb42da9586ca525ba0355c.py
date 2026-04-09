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
    PP = 5
    start_hour = 7
    end_hour = 10
    end_minute = 59

    open_series = df['open'].copy()
    high_series = df['high'].copy()
    low_series = df['low'].copy()
    close_series = df['close'].copy()
    time_series = df['time'].copy()

    def wilder_rsi(prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def wilder_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr

    def pivothigh(source, left_len, right_len):
        ph = source.rolling(left_len + right_len + 1, center=True).max()
        for i in range(right_len):
            ph = ph.where(source.shift(right_len - i) <= ph, np.nan)
        return ph

    def pivotlow(source, left_len, right_len):
        pl = source.rolling(left_len + right_len + 1, center=True).min()
        for i in range(right_len):
            pl = pl.where(source.shift(right_len - i) >= pl, np.nan)
        return pl

    HighPivot = pivothigh(high_series, PP, PP)
    LowPivot = pivotlow(low_series, PP, PP)

    HighPivot = HighPivot.fillna(0)
    LowPivot = LowPivot.fillna(0)
    HighPivot = HighPivot.replace(0, np.nan)
    LowPivot = LowPivot.replace(0, np.nan)

    HighValue = high_series.copy()
    LowValue = low_series.copy()
    HighIndex = np.arange(len(df))
    LowIndex = np.arange(len(df))

    Major_HighLevel = np.nan
    Major_LowLevel = np.nan
    Major_HighIndex = np.nan
    Major_LowIndex = np.nan
    Major_HighType = ""
    Major_LowType = ""

    LastMHH = 0
    Last02MHH = 0
    LastMLH = 0
    LastMLL = 0
    Last02MLL = 0
    LastMHL = 0
    LastmHH = 0
    Last02mHH = 0
    LastmLH = 0
    LastmLL = 0
    Last02mLL = 0
    LastmHL = 0
    LastPivotType = ""
    LastPivotIndex = 0
    LastPivotType02 = ""
    LastPivotIndex02 = 0

    Bullish_Major_ChoCh = False
    Bullish_Major_BoS = False
    Bearish_Major_ChoCh = False
    Bearish_Major_BoS = False

    entries = []
    trade_num = 0

    for i in range(PP * 2 + 5, len(df)):
        ts = int(time_series.iloc[i])
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute

        in_trading_window = (hour >= start_hour and hour <= end_hour) and not (hour == end_hour and minute > end_minute)

        if pd.isna(HighPivot.iloc[i]) and pd.isna(LowPivot.iloc[i]):
            continue

        MajorHighValue01 = Major_HighLevel
        MajorHighIndex01 = int(Major_HighIndex) if not pd.isna(Major_HighIndex) else 0
        MajorLowValue01 = Major_LowLevel
        MajorLowIndex01 = int(Major_LowIndex) if not pd.isna(Major_LowIndex) else 0

        if pd.notna(HighPivot.iloc[i]):
            Correct_HighPivot_val = high_series.iloc[i]
            is_higher_high = False
            is_lower_high = False

            if pd.notna(Major_HighLevel):
                if high_series.iloc[i] > Major_HighLevel:
                    is_higher_high = True
                elif high_series.iloc[i] < Major_HighLevel:
                    is_lower_high = True

            if is_higher_high:
                Last02MHH = LastMHH
                LastMHH = LastPivotIndex
                Major_HighLevel = high_series.iloc[i]
                Major_HighIndex = i
                Major_HighType = "HH"
            elif is_lower_high:
                LastMLH = LastPivotIndex
                Major_HighLevel = high_series.iloc[i]
                Major_HighIndex = i
                Major_HighType = "LH"

        if pd.notna(LowPivot.iloc[i]):
            Correct_LowPivot_val = low_series.iloc[i]
            is_higher_low = False
            is_lower_low = False

            if pd.notna(Major_LowLevel):
                if low_series.iloc[i] > Major_LowLevel:
                    is_higher_low = True
                elif low_series.iloc[i] < Major_LowLevel:
                    is_lower_low = True

            if is_higher_low:
                Last02MLL = LastMLL
                LastMLL = LastPivotIndex
                Major_LowLevel = low_series.iloc[i]
                Major_LowIndex = i
                Major_LowType = "LL"
            elif is_lower_low:
                LastMHL = LastPivotIndex
                Major_LowLevel = low_series.iloc[i]
                Major_LowIndex = i
                Major_LowType = "HL"

        if pd.notna(Major_HighLevel) and pd.notna(Major_LowLevel):
            if LastMHH > 0 and LastMLL > 0 and LastMHH > LastMLL:
                Bullish_Major_BoS = True
            if LastMLH > 0 and LastMLL > 0 and LastMLH > LastMLL:
                Bullish_Major_ChoCh = True
            if LastMHH > 0 and LastMLL > 0 and LastMLL > LastMHH:
                Bearish_Major_BoS = True
            if LastMHH > 0 and LastMLH > 0 and LastMHL > LastMLH:
                Bearish_Major_ChoCh = True

        if pd.notna(HighPivot.iloc[i]):
            LastPivotType = "H"
            LastPivotIndex = i

        if pd.notna(LowPivot.iloc[i]):
            LastPivotType = "L"
            LastPivotIndex = i

        entry_signal_long = Bullish_Major_BoS or Bullish_Major_ChoCh
        entry_signal_short = Bearish_Major_BoS or Bearish_Major_ChoCh

        if entry_signal_long and in_trading_window:
            trade_num += 1
            entry_price = close_series.iloc[i]
            entry_time_str = dt.isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

        if entry_signal_short and in_trading_window:
            trade_num += 1
            entry_price = close_series.iloc[i]
            entry_time_str = dt.isoformat()
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': ts,
                'entry_time': entry_time_str,
                'entry_price_guess': entry_price,
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': entry_price,
                'raw_price_b': entry_price
            })

    return entries