import pandas as pd
import numpy as np
from datetime import datetime, timezone

def generate_entries(df: pd.DataFrame) -> list:
    df = df.copy()
    df['time_dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Resample to 4H candles
    ohlc_4h = df.set_index('time_dt').resample('240T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    high_4h = ohlc_4h['high']
    low_4h = ohlc_4h['low']
    close_4h = ohlc_4h['close']
    volume_4h = ohlc_4h['volume']
    
    # Volume filter: volume[1] > sma(volume, 9)[1] * 1.5
    vol_sma_9 = volume_4h.rolling(9).mean()
    vol_filt = volume_4h.shift(1) > vol_sma_9.shift(1) * 1.5
    
    # ATR filter: (low - high[2] > atr) or (low[2] - high > atr)
    tr1 = high_4h - low_4h
    tr2 = (high_4h - close_4h.shift(1)).abs()
    tr3 = (low_4h - close_4h.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20 = tr.ewm(alpha=1/20, adjust=False).mean()
    atr_20_shifted = atr_20.shift(1)
    atr_thresh = atr_20_shifted / 1.5
    bfvg_gap = low_4h - high_4h.shift(2)
    sfvg_gap = high_4h - low_4h.shift(2)
    atrfilt = (bfvg_gap > atr_thresh) | (sfvg_gap > atr_thresh)
    
    # Bullish FVG: low > high[2]
    bfvg1 = (low_4h > high_4h.shift(2)) & vol_filt.fillna(False) & atrfilt.fillna(False)
    # Bearish FVG: high < low[2]
    sfvg1 = (high_4h < low_4h.shift(2)) & vol_filt.fillna(False) & atrfilt.fillna(False)
    
    # Trend filter: sma(close, 54) > sma(close, 54)[1]
    sma_54 = close_4h.rolling(54).mean()
    loc21 = sma_54 > sma_54.shift(1)
    locfiltb = loc21.fillna(False)
    locfilts = (~loc21).fillna(False)
    
    bfvg = bfvg1 & locfiltb
    sfvg = sfvg1 & locfilts
    
    # Get daily open for direction filter
    daily = df.set_index('time_dt').resample('D').agg({'open': 'first'}).dropna()
    daily['date'] = daily.index.date
    df['_date'] = df['time_dt'].dt.date
    merged_daily = df.merge(daily[['open']].rename(columns={'open': 'daily_open'}), 
                           left_on='_date', right_index=True, how='left')
    is_daily_red = merged_daily['daily_open'] > df['close']
    is_daily_green = df['close'] > merged_daily['daily_open']
    bullish_allowed = is_daily_red
    bearish_allowed = is_daily_green
    
    # Detect new 4H candles
    hours = ohlc_4h.index.hour
    is_new_4h_candle = (hours % 4 == 0) & (ohlc_4h.index.minute == 0)
    
    # Detect first bar of day
    df['_prev_date'] = df['_date'].shift(1).bfill()
    first_bar_of_day = df['_date'] != df['_prev_date']
    
    entries = []
    trade_num = 0
    lastFVG = 0
    
    for i in range(2, len(ohlc_4h)):
        ts_4h = ohl