#!/usr/bin/env python3
"""
cached_backtest.py — Pre-fetch all bars once, then run strategies against cache.
Steps:
  1. Fetch all 5 symbols × 4 timeframes bars (once, ~5 min)
  2. Save to cache as parquet (fast to load)
  3. Run each strategy against cached bars (~5s per strategy)

Expected: 1,277 strategies in ~2-3 hours total (vs 25+ hours with API calls per strategy)
"""
import os, sys, json, time, subprocess, csv, pickle, signal
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = '/home/guiuser/.openclaw/workspace'
TRANSLATED_DIR = f'{SCRIPT_DIR}/pine_translated'
CSV_FILE = f'{SCRIPT_DIR}/strategy_stats.csv'
CACHE_FILE = '/tmp/bars_cache.pkl'
VENV_PY = f'{SCRIPT_DIR}/.venv/bin/python'
MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 8))
TIMEOUT = int(os.environ.get('TIMEOUT', 120))

with open(f'{SCRIPT_DIR}/symbols_timeframes.json') as f:
    cfg = json.load(f)
SYMBOLS = cfg['symbols']
TIMEFRAMES = cfg['timeframes']

print(f'[cached] Symbols: {SYMBOLS}')
print(f'[cached] Timeframes: {TIMEFRAMES}')
print(f'[cached] Workers: {MAX_WORKERS}')


# ── Step 1: Fetch & cache bars ────────────────────────────────────────────────

def fetch_all_bars():
    """Fetch all bars, cache to disk, return dict."""
    if os.path.exists(CACHE_FILE):
        mtime = os.path.getmtime(CACHE_FILE)
        age = time.time() - mtime
        if age < 3600:  # cache less than 1 hour old
            print(f'[cached] Loading bars from cache ({age:.0f}s old)...')
            with open(CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        else:
            print(f'[cached] Cache stale ({age:.0f}s), re-fetching...')
    else:
        print('[cached] No cache found, fetching bars...')

    cache = {}
    for sym in SYMBOLS:
        cache[sym] = {}
        for tf in TIMEFRAMES:
            print(f'  Fetching {sym}/{tf}...', end=' ', flush=True)
            try:
                if sym.upper().startswith('BINANCE:'):
                    bars = _fetch_ccxt(sym, tf)
                else:
                    bars = _fetch_yfinance(sym, tf)
                cache[sym][tf] = bars
                print(f'{len(bars)} bars')
            except Exception as e:
                print(f'ERROR: {e}')
                cache[sym][tf] = []

    print('[cached] Saving cache...')
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    print('[cached] Cache saved.')
    return cache


def _fetch_yfinance(symbol, tf):
    import yfinance as yf
    import pandas as pd
    tf_map = {'15': '5m', '60': '1h', '240': '4h', 'D': '1d'}
    clean = symbol.replace('FX:','').replace('OANDA:','').replace('CAPITALCOM:','')
    clean = {'FX:GBPUSD': 'GBPUSD=X', 'FX:XAUUSD': 'GC=F',
             'FX:US30': '^DJI', 'FX:GER40': '^GDAXI'}.get(symbol, clean)
    tf_str = tf_map.get(str(tf), '1d')
    period = '59d' if str(tf) == '15' else '2y'
    try:
        ticker = yf.Ticker(clean)
        df = ticker.history(period=period, interval=tf_str, auto_adjust=True)
    except Exception:
        return []
    if df.empty:
        return []
    # Determine the datetime column
    dt_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            dt_col = col
            break
    if dt_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            dt_col = df.columns[0]
        else:
            return []
    df = df.rename(columns={dt_col: 'time'})
    df['time'] = pd.to_datetime(df['time'], utc=True).astype('int64') // 10**9
    return df.rename(columns={
        'Open':'open','High':'high','Low':'low',
        'Close':'close','Volume':'volume'
    })[['time','open','high','low','close','volume']].to_dict('records')


def _fetch_ccxt(symbol, tf):
    import ccxt
    tf_map = {'15':'15m','60':'1h','240':'4h','D':'1d'}
    ex = ccxt.binance({'enableRateLimit': True})
    raw = symbol.replace('BINANCE:','').upper()
    # Map common symbols
    sym_map = {
        'BTCUSDT': 'BTC/USDT', 'ETHUSDT': 'ETH/USDT',
        'BNBUSDT': 'BNB/USDT', 'SOLUSDT': 'SOL/USDT'
    }
    market = sym_map.get(raw, raw)
    if '/' not in market:
        market += '/USDT'
    tf_str = tf_map.get(str(tf), '1h')
    since = int((time.time() - 2*365*24*3600) * 1000)
    try:
        bars = ex.fetch_ohlcv(market, tf_str, since=since, limit=2000)
        return [{'time': b[0]//1000, 'open':b[1],'high':b[2],'low':b[3],'close':b[4],'volume':b[5]} for b in bars]
    except Exception:
        return []


# ── Step 2: Build worker script (inline) ───────────────────────────────────────

WORKER_TEMPLATE = """#!/usr/bin/env python3
import sys, os, json, signal, pickle
sys.path.insert(0, "%s")
signal.alarm(%d)

# Load cached bars
with open("%s", "rb") as f:
    CACHE = pickle.load(f)

try:
    name, slug, py_path = sys.argv[1:]
    
    import pandas as pd
    from simulate_standardized import compute_atr, simulate_trade, compute_metrics
    
    # Load entry function
    import importlib.util
    spec = importlib.util.spec_from_file_location("s", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "generate_entries", None)
    if not fn:
        sys.exit(0)
    
    syms = %s
    tfs = %s
    results = []
    
    for sym in syms:
        for tf in tfs:
            bars = CACHE.get(sym, {}).get(str(tf), [])
            if not bars:
                continue
            try:
                df = pd.DataFrame(bars)
                atr = compute_atr(bars)
                entries = fn(df)
                if not entries:
                    continue
                sim_trades = []
                for entry in entries:
                    try:
                        sim = simulate_trade(entry, bars, atr)
                        if sim:
                            sim_trades.append(sim)
                    except:
                        pass
                metrics = compute_metrics(sim_trades)
                results.append({
                    "strategy_name": name,
                    "symbol": sym,
                    "timeframe": str(tf),
                    "win_rate": metrics.get("win_rate", 0),
                    "profit_factor": metrics.get("profit_factor", 0),
                    "max_drawdown": metrics.get("max_drawdown", 0),
                    "trades": metrics.get("trades", 0),
                    "reach_1r_count": metrics.get("reach_1r_count", 0),
                    "reach_2r_count": metrics.get("reach_2r_count", 0),
                    "reach_3r_count": metrics.get("reach_3r_count", 0),
                })
            except:
                pass
    
    if results:
        print(json.dumps(results))
    
except Exception as e:
    sys.exit(0)
""" % (SCRIPT_DIR, TIMEOUT, CACHE_FILE, json.dumps(SYMBOLS), json.dumps(TIMEFRAMES))

WORKER_FILE = '/tmp/cached_worker.py'
with open(WORKER_FILE, 'w') as f:
    f.write(WORKER_TEMPLATE)
os.chmod(WORKER_FILE, 0o755)


# ── Step 3: Collect work ───────────────────────────────────────────────────────

def slugify_id(sid):
    return sid.replace(';','_') if sid else ''

with open(f'{SCRIPT_DIR}/classified_strategies.json') as f:
    classified = json.load(f)

strategies = []
seen = {}
for s in classified:
    sid = slugify_id(s.get('id',''))
    ns = slugify_id(s['name'])
    name = s['name']
    if name in seen:
        continue
    slug = sid if sid else ns
    py = f'{TRANSLATED_DIR}/{slug}.py'
    if os.path.exists(py):
        seen[name] = (name, slug, py)
strategies = list(seen.values())
print(f'[cached] {len(strategies)} strategies')

# Done combos
done = set()
if os.path.exists(CSV_FILE):
    with open(CSV_FILE) as f:
        for r in csv.DictReader(f):
            if r['symbol'] in SYMBOLS and r['timeframe'] in TIMEFRAMES:
                done.add((r['strategy_name'], r['symbol'], r['timeframe']))
print(f'[cached] Already done: {len(done)} combos')

# ── Step 1b: Fetch & cache bars ───────────────────────────────────────────
cache = fetch_all_bars()

needed = [s for s in strategies
      if not all((s[0], sym, tf) in done for sym in SYMBOLS for tf in TIMEFRAMES)]
# Deduplicate by name
seen_names = set()
needed_unique = []
for s in needed:
    if s[0] not in seen_names:
        seen_names.add(s[0])
        needed_unique.append(s)
needed = needed_unique
print(f'[cached] Strategies needing work: {len(needed)}')
if not needed:
    print('[cached] All done.')
    sys.exit(0)


# ── Step 4: Run ──────────────────────────────────────────────────────────────

print('[cached] Starting workers...')
start = time.time()
done_count = [0, 0]  # [ok, skip]

def run_one(args):
    name, slug, py_path = args
    try:
        proc = subprocess.run(
            [VENV_PY, WORKER_FILE, name, slug, py_path],
            capture_output=True, text=True, timeout=TIMEOUT + 10,
            cwd=SCRIPT_DIR
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout.strip())
    except:
        pass
    return []

with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(run_one, s): s for s in needed}
    for future in as_completed(futures):
        name, slug, py_path = futures[future]
        try:
            results = future.result()
            done_count[0] += 1
            ds, df = done_count
            elapsed = time.time() - start
            rate = ds / (elapsed/60) if elapsed > 5 else 0
            eta = (len(needed)-ds)/rate if rate > 0 else 0

            if results:
                with open(CSV_FILE, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=[
                        'strategy_name','symbol','timeframe',
                        'win_rate','profit_factor','max_drawdown',
                        'trades','reach_1r_count','reach_2r_count','reach_3r_count'
                    ])
                    for r in results:
                        w.writerow(r)
                print(f'  [{ds:4d}/{len(needed)}] {name[:45]:<45} OK {len(results)} combos, {rate:.1f}/min, ETA {eta:.0f}min')
            else:
                done_count[1] += 1
                print(f'  [{ds:4d}/{len(needed)}] {name[:45]:<45} SKIP')

            sys.stdout.flush()
        except Exception as e:
            print(f'  ERROR: {str(e)[:80]}')

total = time.time() - start
ok, skip = done_count
print(f'\n[cached] Done. {ok} OK, {skip} skipped. {total/60:.1f} min total.')