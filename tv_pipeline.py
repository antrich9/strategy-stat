#!/usr/bin/env python3
"""
tv_pipeline.py — Classify Pine strategies and run standardized ATR/trailing-R backtests.

Commands:
    python3 tv_pipeline.py classify          # scan all scripts, keep real strategies
    python3 tv_pipeline.py backtest          # run backtests on classified strategies
    python3 tv_pipeline.py run-all           # classify + backtest

Outputs:
    classified_strategies.json  — list of real strategy names
    strategy_stats.csv          — one row per (strategy, symbol, timeframe)

Requirements:
    - tv_mcp_config.json        — MCP server connection config
    - symbols_timeframes.json   — which symbols/timeframes to test
    - TradingView Desktop must be running with the MCP server active
"""

import argparse
import csv
import json
import os
import sys
import time

# Import the MCP client from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tv_mcp_client import make_client

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIED_FILE = os.path.join(SCRIPT_DIR, "classified_strategies.json")
STATS_FILE = os.path.join(SCRIPT_DIR, "strategy_stats.csv")
SYMBOLS_FILE = os.path.join(SCRIPT_DIR, "symbols_timeframes.json")

CSV_HEADER = [
    "strategy_name", "symbol", "timeframe",
    "win_rate", "profit_factor", "max_drawdown", "trades",
    "reach_1r_count", "reach_2r_count", "reach_3r_count",
]

# Pine code block appended to each strategy for R-tracking.
# Uses underscore-prefixed names to avoid collisions.
R_TRACKING_PINE = """
// ---- R-TRACKING INJECTED BY tv_pipeline.py ----
var float _rvATR = na
var float _rvEntry = na
var int _rvDir = 0
var int _rv1R = 0
var int _rv2R = 0
var int _rv3R = 0
var bool _rvHit1 = false
var bool _rvHit2 = false
var bool _rvHit3 = false

_rvAtr14 = ta.atr(14)

if strategy.position_size != 0 and strategy.position_size[1] == 0
    _rvATR := _rvAtr14
    _rvEntry := strategy.position_avg_price
    _rvDir := strategy.position_size > 0 ? 1 : -1
    _rvHit1 := false
    _rvHit2 := false
    _rvHit3 := false

if strategy.position_size != 0 and not na(_rvATR)
    _rvStop = _rvATR * 3.0
    if _rvDir == 1
        if not _rvHit1 and high >= _rvEntry + _rvStop * 1.0
            _rvHit1 := true
            _rv1R += 1
        if not _rvHit2 and high >= _rvEntry + _rvStop * 2.0
            _rvHit2 := true
            _rv2R += 1
        if not _rvHit3 and high >= _rvEntry + _rvStop * 3.0
            _rvHit3 := true
            _rv3R += 1
    else
        if not _rvHit1 and low <= _rvEntry - _rvStop * 1.0
            _rvHit1 := true
            _rv1R += 1
        if not _rvHit2 and low <= _rvEntry - _rvStop * 2.0
            _rvHit2 := true
            _rv2R += 1
        if not _rvHit3 and low <= _rvEntry - _rvStop * 3.0
            _rvHit3 := true
            _rv3R += 1

var table _rvTbl = table.new(position.bottom_right, 2, 4, border_width=1)
if barstate.islast
    table.cell(_rvTbl, 0, 0, "R Milestone", bgcolor=color.new(color.blue, 70), text_color=color.white)
    table.cell(_rvTbl, 1, 0, "Count", bgcolor=color.new(color.blue, 70), text_color=color.white)
    table.cell(_rvTbl, 0, 1, "1R reached")
    table.cell(_rvTbl, 1, 1, str.tostring(_rv1R))
    table.cell(_rvTbl, 0, 2, "2R reached")
    table.cell(_rvTbl, 1, 2, str.tostring(_rv2R))
    table.cell(_rvTbl, 0, 3, "3R reached")
    table.cell(_rvTbl, 1, 3, str.tostring(_rv3R))
// ---- END R-TRACKING ----
"""


def is_real_strategy(source: str) -> bool:
    tokens = ["strategy(", "strategy.entry", "strategy.exit", "strategy.close"]
    return any(t in source for t in tokens)


def load_symbols_timeframes():
    with open(SYMBOLS_FILE) as f:
        data = json.load(f)
    return data["symbols"], data["timeframes"]


def load_csv_index():
    """Return dict keyed by (strategy_name, symbol, timeframe) → row dict."""
    index = {}
    if not os.path.exists(STATS_FILE):
        return index
    with open(STATS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["strategy_name"], row["symbol"], row["timeframe"])
            index[key] = row
    return index


def write_csv(index: dict):
    rows = sorted(index.values(), key=lambda r: (r["strategy_name"], r["symbol"], r["timeframe"]))
    with open(STATS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def parse_strategy_results(result: dict):
    """Extract metrics from data_get_strategy_results response."""
    perf = result.get("performance", result)  # handle nested or flat
    # TradingView returns keys like netProfit, winRate, profitFactor, maxDrawdown, totalTrades
    win_rate = (
        perf.get("winRate")
        or perf.get("win_rate")
        or perf.get("percentProfitable")
        or 0.0
    )
    profit_factor = perf.get("profitFactor") or perf.get("profit_factor") or 0.0
    max_drawdown = (
        perf.get("maxDrawdown")
        or perf.get("max_drawdown")
        or perf.get("maxDrawdownPercent")
        or 0.0
    )
    trades = (
        perf.get("totalTrades")
        or perf.get("trades")
        or perf.get("total_trades")
        or 0
    )
    # If win_rate is a fraction (0-1), convert to percent
    if isinstance(win_rate, (int, float)) and 0 < win_rate <= 1.0:
        win_rate = round(win_rate * 100, 2)
    return {
        "win_rate": round(float(win_rate), 2) if win_rate else 0.0,
        "profit_factor": round(float(profit_factor), 4) if profit_factor else 0.0,
        "max_drawdown": round(float(max_drawdown), 4) if max_drawdown else 0.0,
        "trades": int(trades) if trades else 0,
    }


def parse_r_counts(table_result: dict):
    """Parse R-milestone counts from data_get_pine_tables output."""
    r1 = r2 = r3 = 0
    tables = table_result.get("tables", [])
    for tbl in tables:
        rows = tbl.get("rows", [])
        for row in rows:
            cells = row if isinstance(row, list) else []
            if len(cells) >= 2:
                label = str(cells[0]).lower()
                try:
                    val = int(float(str(cells[1])))
                except (ValueError, TypeError):
                    continue
                if "1r" in label:
                    r1 = val
                elif "2r" in label:
                    r2 = val
                elif "3r" in label:
                    r3 = val
    return r1, r2, r3


def wait_for_strategy_tester(client, retries=8, delay=3.0):
    """Poll strategy results until trades > 0 or retries exhausted."""
    for _ in range(retries):
        try:
            result = client.call_tool("data_get_strategy_results", {}, timeout=30)
            if result.get("success") is False:
                time.sleep(delay)
                continue
            metrics = parse_strategy_results(result)
            if metrics["trades"] > 0:
                return result
        except Exception:
            pass
        time.sleep(delay)
    return None


# ---------------------------------------------------------------------------
# CLASSIFY
# ---------------------------------------------------------------------------

def ensure_pine_editor_open(client):
    """Open the Pine editor panel — required for pine_open and pine_get_source."""
    try:
        result = client.call_tool("ui_open_panel", {"panel": "pine-editor", "action": "open"}, timeout=15)
        if result.get("performed") == "opened":
            time.sleep(1.0)  # let the panel fully load
    except Exception as e:
        print(f"  [warn] Could not open Pine editor panel: {e}")


def cmd_classify():
    print("[classify] Connecting to TradingView MCP...")
    client = make_client()
    try:
        print("[classify] Opening Pine editor panel...")
        ensure_pine_editor_open(client)
        print("[classify] Listing Pine scripts...")
        result = client.call_tool("pine_list_scripts", {}, timeout=30)
        if result.get("success") is False:
            print(f"[classify] ERROR: {result.get('error')}", file=sys.stderr)
            sys.exit(1)

        scripts = result.get("scripts", [])
        if not scripts:
            # Try alternative key names
            scripts = result.get("items", result.get("list", []))
        print(f"[classify] Found {len(scripts)} saved scripts.")

        strategies = []
        failed = []

        for i, script in enumerate(scripts):
            # pine_list_scripts returns both 'name' (internal) and 'title' (display name)
            # pine_open does case-insensitive match on the display name (title)
            name = script.get("title") or script.get("name") or str(script)
            print(f"[classify] [{i+1}/{len(scripts)}] Checking: {name}")

            try:
                # Open the script to load it into the editor
                client.call_tool("pine_open", {"name": name}, timeout=30)
                time.sleep(0.5)

                # Get source
                src_result = client.call_tool("pine_get_source", {}, timeout=30)
                source = src_result.get("source") or src_result.get("code") or ""

                if not source:
                    print(f"  → No source retrieved, skipping.")
                    failed.append(name)
                    continue

                if is_real_strategy(source):
                    print(f"  → STRATEGY (has strategy() calls)")
                    strategies.append({"name": name, "id": script.get("id", "")})
                else:
                    print(f"  → indicator, skip")

            except Exception as e:
                print(f"  → ERROR: {e}, skipping")
                failed.append(name)

            # Rate limit: small delay between scripts
            if (i + 1) % 25 == 0:
                print(f"[classify] Batch pause...")
                time.sleep(1.0)

    finally:
        client.stop()

    with open(CLASSIFIED_FILE, "w") as f:
        json.dump(strategies, f, indent=2)

    print(f"\n[classify] Done. {len(strategies)} real strategies saved to {CLASSIFIED_FILE}")
    if failed:
        print(f"[classify] {len(failed)} scripts skipped due to errors.")


# ---------------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------------

def cmd_backtest(start=0, limit=None, symbols_override=None, timeframes_override=None, skip_existing=False):
    if not os.path.exists(CLASSIFIED_FILE):
        print(f"[backtest] ERROR: {CLASSIFIED_FILE} not found. Run classify first.", file=sys.stderr)
        sys.exit(1)

    with open(CLASSIFIED_FILE) as f:
        strategies = json.load(f)

    total_classified = len(strategies)
    if start < 0:
        start = 0
    if start >= total_classified:
        print(f"[backtest] ERROR: start={start} is beyond classified strategy count ({total_classified}).", file=sys.stderr)
        sys.exit(1)

    end = total_classified if limit is None else min(total_classified, start + max(limit, 0))
    strategies = strategies[start:end]

    file_symbols, file_timeframes = load_symbols_timeframes()
    symbols = symbols_override if symbols_override else file_symbols
    timeframes = timeframes_override if timeframes_override else file_timeframes
    csv_index = load_csv_index()

    total = len(strategies) * len(symbols) * len(timeframes)
    done = 0
    original_source = None

    print(
        f"[backtest] Selected {len(strategies)} strategies (from {total_classified}, start={start}, end={end}) × "
        f"{len(symbols)} symbols × {len(timeframes)} timeframes = {total} combinations"
    )
    if skip_existing:
        print(f"[backtest] skip-existing enabled; current CSV rows: {len(csv_index)}")
    print("[backtest] Connecting to TradingView MCP...")

    client = make_client()
    try:
        print("[backtest] Opening Pine editor panel...")
        ensure_pine_editor_open(client)
        for strat_i, strat in enumerate(strategies, start=1):
            name = strat["name"]
            print(f"\n[backtest] Strategy [{strat_i}/{len(strategies)} | global {start + strat_i}/{total_classified}]: {name}")

            # Open and get original source (pine_open does case-insensitive match on title)
            try:
                client.call_tool("pine_open", {"name": name}, timeout=30)
                time.sleep(0.8)
                src_result = client.call_tool("pine_get_source", {}, timeout=30)
                original_source = src_result.get("source") or src_result.get("code") or ""
            except Exception as e:
                print(f"  ERROR opening script: {e}, skipping strategy")
                continue

            if not original_source:
                print("  No source, skipping")
                continue

            # Inject R-tracking code
            injected_source = original_source + "\n" + R_TRACKING_PINE

            for symbol in symbols:
                for tf in timeframes:
                    key = (name, symbol, tf)
                    done += 1

                    if skip_existing and key in csv_index:
                        print(f"  [{done}/{total}] {symbol} / {tf} ... skip-existing")
                        continue

                    print(f"  [{done}/{total}] {symbol} / {tf}", end=" ... ", flush=True)

                    try:
                        # Set symbol and timeframe
                        client.call_tool("chart_set_symbol", {"symbol": symbol}, timeout=30)
                        time.sleep(0.5)
                        client.call_tool("chart_set_timeframe", {"timeframe": tf}, timeout=30)
                        time.sleep(1.0)

                        # Inject R-tracking source and compile
                        client.call_tool("pine_set_source", {"source": injected_source}, timeout=30)
                        time.sleep(0.3)
                        compile_result = client.call_tool("pine_smart_compile", {}, timeout=60)
                        if not compile_result.get("success", True) is False:
                            # Check for compile errors
                            errors = compile_result.get("errors", [])
                            if errors:
                                # Fall back to original source without R-tracking
                                print(f"compile errors ({len(errors)}), using original source")
                                client.call_tool("pine_set_source", {"source": original_source}, timeout=30)
                                client.call_tool("pine_smart_compile", {}, timeout=60)
                                time.sleep(2.0)
                                use_r_tracking = False
                            else:
                                time.sleep(3.0)  # Let strategy tester run
                                use_r_tracking = True
                        else:
                            use_r_tracking = True
                            time.sleep(3.0)

                        # Get strategy results
                        sr = wait_for_strategy_tester(client)
                        if sr is None:
                            print("no strategy results, skipping")
                            continue

                        metrics = parse_strategy_results(sr)

                        # Get R counts from the injected table
                        r1 = r2 = r3 = 0
                        if use_r_tracking:
                            try:
                                table_result = client.call_tool("data_get_pine_tables", {}, timeout=30)
                                r1, r2, r3 = parse_r_counts(table_result)
                            except Exception:
                                pass

                        row = {
                            "strategy_name": name,
                            "symbol": symbol,
                            "timeframe": tf,
                            "win_rate": metrics["win_rate"],
                            "profit_factor": metrics["profit_factor"],
                            "max_drawdown": metrics["max_drawdown"],
                            "trades": metrics["trades"],
                            "reach_1r_count": r1,
                            "reach_2r_count": r2,
                            "reach_3r_count": r3,
                        }
                        csv_index[key] = row
                        write_csv(csv_index)

                        print(f"OK (trades={metrics['trades']}, wr={metrics['win_rate']}%, 1R={r1} 2R={r2} 3R={r3})")

                    except Exception as e:
                        print(f"ERROR: {e}")

                    # Small delay between combos; batch pause every 25
                    time.sleep(0.3)
                    if done % 25 == 0:
                        print("  [batch pause]")
                        time.sleep(1.5)

    finally:
        # Restore original source for last strategy
        if original_source:
            try:
                client.call_tool("pine_set_source", {"source": original_source}, timeout=30)
                client.call_tool("pine_smart_compile", {}, timeout=30)
            except Exception:
                pass
        client.stop()

    print(f"\n[backtest] Done. Results written to {STATS_FILE}")
    print(f"[backtest] Total rows in CSV: {len(csv_index)}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def parse_cli():
    parser = argparse.ArgumentParser(description="Classify Pine strategies and run standardized ATR/trailing-R backtests.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    subparsers.add_parser("classify", help="scan all scripts and keep real strategies")

    backtest_parser = subparsers.add_parser("backtest", help="run backtests on classified strategies")
    backtest_parser.add_argument("--start", type=int, default=0, help="0-based strategy offset in classified_strategies.json")
    backtest_parser.add_argument("--limit", type=int, default=None, help="maximum number of strategies to backtest")
    backtest_parser.add_argument("--symbols", type=str, default=None, help="comma-separated symbol override")
    backtest_parser.add_argument("--timeframes", type=str, default=None, help="comma-separated timeframe override")
    backtest_parser.add_argument("--skip-existing", action="store_true", help="skip rows already present in strategy_stats.csv")

    run_all_parser = subparsers.add_parser("run-all", help="classify and then backtest")
    run_all_parser.add_argument("--start", type=int, default=0, help="0-based strategy offset in classified_strategies.json")
    run_all_parser.add_argument("--limit", type=int, default=None, help="maximum number of strategies to backtest")
    run_all_parser.add_argument("--symbols", type=str, default=None, help="comma-separated symbol override")
    run_all_parser.add_argument("--timeframes", type=str, default=None, help="comma-separated timeframe override")
    run_all_parser.add_argument("--skip-existing", action="store_true", help="skip rows already present in strategy_stats.csv")

    return parser.parse_args()


def split_csv_arg(value):
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    args = parse_cli()

    if args.cmd == "classify":
        cmd_classify()
    elif args.cmd == "backtest":
        cmd_backtest(
            start=args.start,
            limit=args.limit,
            symbols_override=split_csv_arg(args.symbols),
            timeframes_override=split_csv_arg(args.timeframes),
            skip_existing=args.skip_existing,
        )
    elif args.cmd == "run-all":
        cmd_classify()
        cmd_backtest(
            start=args.start,
            limit=args.limit,
            symbols_override=split_csv_arg(args.symbols),
            timeframes_override=split_csv_arg(args.timeframes),
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
