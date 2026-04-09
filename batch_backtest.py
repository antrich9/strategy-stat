#!/usr/bin/env python3
"""
batch_backtest.py — Batch all symbol/timeframe combos per strategy into a single call.
Each strategy runs as ONE subprocess; all 5 symbols × 4 TFs at once.
8 workers = 8 strategies running simultaneously.

Usage:
  .venv/bin/python batch_backtest.py                  # 8 workers
  MAX_WORKERS=4 .venv/bin/python batch_backtest.py   # 4 workers
"""
import os, sys, json, time, subprocess, csv, signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

SCRIPT_DIR = '/home/guiuser/.openclaw/workspace'
TRANSLATED_DIR = f'{SCRIPT_DIR}/pine_translated'
CSV_FILE = f'{SCRIPT_DIR}/strategy_stats.csv'
VENV_PY = f'{SCRIPT_DIR}/.venv/bin/python'

TIMEOUT = int(os.environ.get('TIMEOUT', 120))  # 2 min per strategy (more generous)
MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 8))

with open(f'{SCRIPT_DIR}/symbols_timeframes.json') as f:
    cfg = json.load(f)
SYMBOLS = cfg['symbols']
TIMEFRAMES = cfg['timeframes']

print(f'[batch] Symbols: {SYMBOLS}')
print(f'[batch] TFs: {TIMEFRAMES}')
print(f'[batch] Timeout: {TIMEOUT}s, Workers: {MAX_WORKERS}')


# ── Worker script (inline, one file) ───────────────────────────────────────────

WORKER = """#!/usr/bin/env python3
import sys, os, json, signal
sys.path.insert(0, "%s")
signal.alarm(%d)
try:
    name, slug, py_path = sys.argv[1:]
    
    import pandas as pd
    from simulate_standardized import (
        fetch_bars_via_ccxt, fetch_bars_via_yfinance,
        compute_atr, simulate_trade, compute_metrics
    )
    
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
        # Fetch bars ONCE per symbol
        if sym.upper().startswith("BINANCE:"):
            bars = fetch_bars_via_ccxt(sym, "60")
        else:
            bars = fetch_bars_via_yfinance(sym, "60")
        if not bars:
            continue
        
        df = pd.DataFrame(bars)
        atr = compute_atr(bars)
        
        for tf in tfs:
            try:
                # Re-fetch per timeframe (skip for 60 since already have it)
                if str(tf) == "60":
                    bars_tf = bars
                elif sym.upper().startswith("BINANCE:"):
                    bars_tf = fetch_bars_via_ccxt(sym, str(tf))
                else:
                    bars_tf = fetch_bars_via_yfinance(sym, str(tf))
                if not bars_tf:
                    continue
                
                df_tf = pd.DataFrame(bars_tf)
                atr_tf = compute_atr(bars_tf)
                entries = fn(df_tf)
                if not entries:
                    continue
                
                sim_trades = []
                for entry in entries:
                    try:
                        sim = simulate_trade(entry, bars_tf, atr_tf)
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
""" % (SCRIPT_DIR, TIMEOUT, json.dumps(SYMBOLS), json.dumps(TIMEFRAMES))

WORKER_FILE = '/tmp/batchbt_worker.py'
with open(WORKER_FILE, 'w') as f:
    f.write(WORKER)
os.chmod(WORKER_FILE, 0o755)


# ── Build strategy list ───────────────────────────────────────────────────────

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
print(f'[batch] {len(strategies)} strategies')

done_combos = set()
if os.path.exists(CSV_FILE):
    with open(CSV_FILE) as f:
        for r in csv.DictReader(f):
            if r['symbol'] in SYMBOLS and r['timeframe'] in TIMEFRAMES:
                done_combos.add((r['strategy_name'], r['symbol'], r['timeframe']))
print(f'[batch] Already done: {len(done_combos)} combos')

# Strategies still needed
needed = []
for name, slug, py_path in strategies:
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            if (name, sym, tf) not in done_combos:
                needed.append((name, slug, py_path))
                break  # only need to know strategy is needed, not how many combos

needed = list({n: (n,s,p) for n,s,p in needed}.values())
print(f'[batch] Strategies needing work: {len(needed)}')
if not needed:
    print('[batch] All done.')
    sys.exit(0)


# ── Run one strategy ───────────────────────────────────────────────────────────

def run_strategy(args):
    name, slug, py_path = args
    try:
        proc = subprocess.run(
            [VENV_PY, WORKER_FILE, name, slug, py_path],
            capture_output=True, text=True, timeout=TIMEOUT + 20,
            cwd=SCRIPT_DIR
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout.strip())
    except:
        pass
    return []


# ── Main ──────────────────────────────────────────────────────────────────────

start = time.time()
done = 0

with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(run_strategy, s): s for s in needed}

    for future in as_completed(futures):
        name, slug, py_path = futures[future]
        try:
            results = future.result()
            done += 1
            elapsed = time.time() - start
            rate = done / (elapsed / 60) if elapsed > 5 else 0
            eta = (len(needed) - done) / rate if rate > 0 else 0

            if results:
                with open(CSV_FILE, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=[
                        'strategy_name','symbol','timeframe',
                        'win_rate','profit_factor','max_drawdown',
                        'trades','reach_1r_count','reach_2r_count','reach_3r_count'
                    ])
                    for r in results:
                        w.writerow(r)
                print(f'  [{done:4d}/{len(needed)}] {name[:45]:<45} OK {len(results)} combos, {rate:.1f}/min, ETA {eta:.0f}min')
            else:
                print(f'  [{done:4d}/{len(needed)}] {name[:45]:<45} SKIP')

            sys.stdout.flush()
        except Exception as e:
            print(f'  ERROR: {str(e)[:80]}')

total = time.time() - start
print(f'\n[batch] Done. {done}/{len(needed)} strategies. {total/60:.1f} min total.')