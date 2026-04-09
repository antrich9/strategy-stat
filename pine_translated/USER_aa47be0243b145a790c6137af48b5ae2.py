import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    # SSL2 parameters
    SSL2_len = 5
    SSL2_beta = 0.8
    SSL2_feedback = False
    SSL2_z = 0.5
    
    # ATR parameters
    ATR_len = 14
    ATR_mult = 1.0
    
    # Baseline parameters
    HMA_len = 60
    multy = 0.2
    Kelt_len = 60
    
    # Jurik parameters for SSL2
    jurik_phase = 3
    jurik_power = 1
    
    def jurik_ma(src, length):
        phase_ratio = 0.5 if jurik_phase < -100 else 2.5 if jurik_phase > 100 else jurik_phase / 100 + 1.5
        beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
        alpha = np.power(beta, jurik_power)
        result = np.zeros(len(src))
        e0 = np.zeros(len(src))
        e1 = np.zeros(len(src))
        e2 = np.zeros(len(src))
        for i in range(1, len(src)):
            e0[i] = (1 - alpha) * src[i] + alpha * e0[i-1] if not np.isnan(e0[i-1]) else src[i]
            e1[i] = (src[i] - e0[i]) * (1 - beta) + beta * e1[i-1] if not np.isnan(e1[i-1]) else 0
            e2[i] = (e0[i] + phase_ratio * e1[i] - result[i-1]) * np.power(1 - alpha, 2) + np.power(alpha, 2) * e2[i-1] if not np.isnan(e2[i-1]) else 0
            result[i] = e2[i] + result[i-1] if not np.isnan(result[i-1]) else e2[i]
        return result
    
    def hma(src, length):
        half = length // 2
        sqrt_len = int(np.sqrt(length))
        wma1 = pd.Series(src).rolling(half).apply(lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True).values
        wma2 = pd.Series(src).rolling(length).apply(lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True).values
        hma2 = 2 * wma1 - wma2
        result = pd.Series(hma2).rolling(sqrt_len).apply(lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True).values
        return result
    
    def wilder_smooth(src, length):
        alpha = 1 / length
        result = np.zeros(len(src))
        result[0] = src[0]
        for i in range(1, len(src)):
            result[i] = alpha * src[i] + (1 - alpha) * result[i-1]
        return result
    
    # Calculate ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr_slen = wilder_smooth(tr, ATR_len)
    upper_band = atr_slen * ATR_mult + close
    lower_band = close - atr_slen * ATR_mult
    
    # Calculate BBMC (HMA baseline)
    BBMC = hma(close, HMA_len)
    
    # Calculate Keltner Channel
    Keltma = hma(close, Kelt_len)
    range_tr = np.zeros(len(df))
    range_tr[0] = tr1[0]
    for i in range(1, len(df)):
        range_tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    rangema = pd.Series(range_tr).ewm(span=Kelt_len, adjust=False).mean().values
    upperk = Keltma + rangema * multy
    lowerk = Keltma - rangema * multy
    
    # Calculate SSL2
    maHigh = jurik_ma(high, SSL2_len)
    maLow = jurik_ma(low, SSL2_len)
    Hlv2 = np.zeros(len(close))
    sslDown2 = np.zeros(len(close))
    for i in range(len(close)):
        if i == 0:
            Hlv2[i] = np.nan
        else:
            if close[i] > maHigh[i]:
                Hlv2[i] = 1
            elif close[i] < maLow[i]:
                Hlv2[i] = -1
            else:
                Hlv2[i] = Hlv2[i-1]
        sslDown2[i] = maHigh[i] if Hlv2[i] < 0 else maLow[i]
    
    # ATR criteria
    atr_crit = 0.9
    upper_half = atr_slen * atr_crit + close
    lower_half = close - atr_slen * atr_crit
    buy_inatr = lower_half < sslDown2
    sell_inatr = upper_half > sslDown2
    sell_cont = (close < BBMC) & (close < sslDown2)
    buy_cont = (close > BBMC) & (close > sslDown2)
    buy_atr = buy_inatr & buy_cont
    sell_atr = sell_inatr & sell_cont
    
    # Baseline crossovers
    baseline_long_cond = pd.Series(close > upperk) & pd.Series(close).shift(1).le(upperk)
    baseline_short_cond = pd.Series(close < lowerk) & pd.Series(close).shift(1).ge(lowerk)
    
    entries = []
    trade_num = 1
    
    for i in range(len(df)):
        if i == 0:
            continue
        
        entry_price = float(close[i])
        ts = int(df['time'].iloc[i])
        entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        
        if baseline_long_cond.iloc[i] or buy_atr[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
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
        
        if baseline_short_cond.iloc[i] or sell_atr[i]:
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
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