import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df.reset_index(drop=True, inplace=True)
    
    # QQE MOD Parameters
    RSI_Period = 6
    SF = 6
    QQE = 3
    Wilders_Period = RSI_Period * 2 - 1
    length = 50
    qqeMult = 0.35
    
    # Calculate RSI (Wilder)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    rs = avg_gain / avg_loss
    Rsi = 100 - (100 / (1 + rs))
    
    RsiMa = Rsi.ewm(span=SF, adjust=False).mean()
    AtrRsi = abs(RsiMa.shift(1) - RsiMa)
    MaAtrRsi = AtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean()
    dar = MaAtrRsi.ewm(alpha=1/Wilders_Period, adjust=False).mean() * QQE
    
    RSIndex = RsiMa
    newshortband = RSIndex + dar
    newlongband = RSIndex - dar
    
    longband = pd.Series(0.0, index=df.index)
    shortband = pd.Series(0.0, index=df.index)
    trend = pd.Series(1, index=df.index)
    
    for i in range(1, len(df)):
        prev_longband = longband.iloc[i-1]
        prev_shortband = shortband.iloc[i-1]
        rs_prev = RSIndex.iloc[i-1]
        rs_curr = RSIndex.iloc[i]
        
        if rs_prev > prev_longband and rs_curr > prev_longband:
            longband.iloc[i] = max(prev_longband, newlongband.iloc[i])
        else:
            longband.iloc[i] = newlongband.iloc[i]
            
        if rs_prev < prev_shortband and rs_curr < prev_shortband:
            shortband.iloc[i] = min(prev_shortband, newshortband.iloc[i])
        else:
            shortband.iloc[i] = newshortband.iloc[i]
        
        cross_short = rs_curr < shortband.iloc[i] and rs_prev >= shortband.iloc[i-1]
        cross_long = rs_curr > longband.iloc[i] and rs_prev <= longband.iloc[i-1]
        
        if cross_long:
            trend.iloc[i] = 1
        elif cross_short:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
    
    FastAtrRsiTL = pd.Series(np.where(trend == 1, longband, shortband), index=df.index)
    
    # Bollinger Bands on QQE
    qqe_center = FastAtrRsiTL - 50
    basis = qqe_center.rolling(length).mean()
    dev = qqeMult * qqe_center.rolling(length).std()
    upper = basis + dev
    lower = basis - dev
    
    # QQE Zero
    QQEzlong = pd.Series(0, index=df.index)
    QQEzshort = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if RSIndex.iloc[i] >= 50:
            QQEzlong.iloc[i] = QQEzlong.iloc[i-1] + 1
            QQEzshort.iloc[i] = 0
        else:
            QQEzshort.iloc[i] = QQEzshort.iloc[i-1] + 1
            QQEzlong.iloc[i] = 0
    
    # QQE MOD 2
    RSI_Period2 = 6
    SF2 = 5
    QQE2 = 1.61
    Wilders_Period2 = RSI_Period2 * 2 - 1
    
    delta2 = df['close'].diff()
    gain2 = delta2.clip(lower=0)
    loss2 = (-delta2).clip(lower=0)
    avg_gain2 = gain2.ewm(alpha=1/Wilders_Period2, adjust=False).mean()
    avg_loss2 = loss2.ewm(alpha=1/Wilders_Period2, adjust=False).mean()
    rs2 = avg_gain2 / avg_loss2
    Rsi2 = 100 - (100 / (1 + rs2))
    
    RsiMa2 = Rsi2.ewm(span=SF2, adjust=False).mean()
    AtrRsi2 = abs(RsiMa2.shift(1) - RsiMa2)
    MaAtrRsi2 = AtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean()
    dar2 = MaAtrRsi2.ewm(alpha=1/Wilders_Period2, adjust=False).mean() * QQE2
    
    RSIndex2 = RsiMa2
    newshortband2 = RSIndex2 + dar2
    newlongband2 = RSIndex2 - dar2
    
    longband2 = pd.Series(0.0, index=df.index)
    shortband2 = pd.Series(0.0, index=df.index)
    trend2 = pd.Series(1, index=df.index)
    
    for i in range(1, len(df)):
        prev_longband2 = longband2.iloc[i-1]
        prev_shortband2 = shortband2.iloc[i-1]
        rs_prev2 = RSIndex2.iloc[i-1]
        rs_curr2 = RSIndex2.iloc[i]
        
        if rs_prev2 > prev_longband2 and rs_curr2 > prev_longband2:
            longband2.iloc[i] = max(prev_longband2, newlongband2.iloc[i])
        else:
            longband2.iloc[i] = newlongband2.iloc[i]
            
        if rs_prev2 < prev_shortband2 and rs_curr2 < prev_shortband2:
            shortband2.iloc[i] = min(prev_shortband2, newshortband2.iloc[i])
        else:
            shortband2.iloc[i] = newshortband2.iloc[i]
        
        cross_short2 = rs_curr2 < shortband2.iloc[i] and rs_prev2 >= shortband2.iloc[i-1]
        cross_long2 = rs_curr2 > longband2.iloc[i] and rs_prev2 <= longband2.iloc[i-1]
        
        if cross_long2:
            trend2.iloc[i] = 1
        elif cross_short2:
            trend2.iloc[i] = -1
        else:
            trend2.iloc[i] = trend2.iloc[i-1]
    
    # SSL Hybrid (simplified using EMA crossovers)
    sslLen = 10
    ema1 = df['close'].ewm(span=sslLen, adjust=False).mean()
    ema2 = df['close'].ewm(span=sslLen * 2, adjust=False).mean()
    sslUp = ema1 > ema2
    sslDown = ema1 < ema2
    
    # Waddah Attar Explosion (simplified)
    deadLength = 20
    explLength = 20
    sensitivity = 20
    deadLevel = 1.5
    explLevel = 3.0
    
    expDiff = df['close'].ewm(span=explLength, adjust=False).mean() - df['close'].ewm(span=deadLength, adjust=False).mean()
    was_short = expDiff < 0
    was_long = expDiff > 0
    
    # On-Chain indicators (calculated from available data)
    # Volume Delta / CVD
    buy_vol = df['volume'] * ((df['close'] - df['low']) / (df['high'] - df['low']).clip(lower=0.0001))
    sell_vol = df['volume'] * ((df['high'] - df['close']) / (df['high'] - df['low']).clip(lower=0.0001))
    vol_delta = buy_vol - sell_vol
    cvd = vol_delta.cumsum()
    cvd_ma = cvd.rolling(14).mean()
    cvd_bullish = cvd > cvd_ma
    cvd_bearish = cvd < cvd_ma
    
    # MFI
    mfi_len = 14
    typical = (df['high'] + df['low'] + df['close']) / 3
    raw_mf = typical * df['volume']
    pos_flow = pd.Series(np.where(typical.diff() > 0, raw_mf, 0), index=df.index)
    neg_flow = pd.Series(np.where(typical.diff() < 0, raw_mf, 0), index=df.index)
    pos_mf = pos_flow.rolling(mfi_len).sum()
    neg_mf = neg_flow.rolling(mfi_len).sum()
    mfi = 100 - (100 / (1 + pos_mf / neg_mf.clip(lower=0.0001)))
    mfi_bullish = (mfi > 50) & (mfi < 80)
    mfi_bearish = (mfi < 50) & (mfi > 20)
    
    # OBV
    obv_change = np.where(df['close'] > df['close'].shift(1), df['volume'],
                          np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
    obv = pd.Series(obv_change, index=df.index).cumsum()
    obv_ma = obv.rolling(20).mean()
    obv_bullish = obv > obv_ma
    obv_bearish = obv < obv_ma
    
    # CMF
    ad = np.where(df['high'] == df['low'], 0,
                  ((2 * df['close'] - df['low'] - df['high']) / (df['high'] - df['low']).clip(lower=0.0001)) * df['volume'])
    ad = pd.Series(ad, index=df.index)
    cmf = ad.rolling(20).mean() / df['volume'].rolling(20).mean()
    cmf_bullish = cmf > 0.05
    cmf_bearish = cmf < -0.05
    
    # Volume filter
    vol_ma = df['volume'].rolling(20).mean()
    high_vol = df['volume'] > vol_ma * 1.5
    
    # VWAP (simplified)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    price_above_vwap = df['close'] > vwap
    price_below_vwap = df['close'] < vwap
    
    # Entry conditions
    long_condition = (trend == 1) & (trend2 == 1) & sslUp & cvd_bullish & mfi_bullish & obv_bullish & cmf_bullish & high_vol & price_above_vwap
    short_condition = (trend == -1) & (trend2 == -1) & sslDown & cvd_bearish & mfi_bearish & obv_bearish & cmf_bearish & high_vol & price_below_vwap
    
    entries = []
    trade_num = 1
    
    for i in range(1, len(df)):
        if long_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'long',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
        elif short_condition.iloc[i]:
            entry_ts = int(df['time'].iloc[i])
            entries.append({
                'trade_num': trade_num,
                'direction': 'short',
                'entry_ts': entry_ts,
                'entry_time': datetime.fromtimestamp(entry_ts, tz=timezone.utc).isoformat(),
                'entry_price_guess': float(df['close'].iloc[i]),
                'exit_ts': 0,
                'exit_time': '',
                'exit_price_guess': 0.0,
                'raw_price_a': float(df['close'].iloc[i]),
                'raw_price_b': float(df['close'].iloc[i])
            })
            trade_num += 1
    
    return entries