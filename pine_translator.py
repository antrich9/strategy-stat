#!/usr/bin/env python3
"""
pine_translator.py — Translate Pine Script entry logic to Python via Claude API.

For each .pine file in pine_scripts/:
  1. Reads the Pine Script source
  2. Calls Claude API (via ANTHROPIC_API_KEY) to extract entry conditions as Python
  3. Validates syntax of returned code
  4. Saves to pine_translated/{slug}.py

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    .venv/bin/python pine_translator.py                     # all captured scripts
    .venv/bin/python pine_translator.py --slug USER_abc123  # single by ID slug
    .venv/bin/python pine_translator.py --limit 20          # first 20 not yet translated
    .venv/bin/python pine_translator.py --force             # re-translate all

Output:
    pine_translated/{slug}.py   — Python generate_entries(df) function
    pine_translation_log.json   — success/failure log
"""

import argparse
import ast
import json
import os
import re
import sys
import time

try:
    import requests
except ImportError:
    raise SystemExit('requests not available: run `pip install requests` in venv')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PINE_DIR = os.path.join(SCRIPT_DIR, 'pine_scripts')
TRANSLATED_DIR = os.path.join(SCRIPT_DIR, 'pine_translated')
CLASSIFIED_FILE = os.path.join(SCRIPT_DIR, 'classified_strategies.json')
LOG_FILE = os.path.join(SCRIPT_DIR, 'pine_translation_log.json')

OPENCLAW_CONFIG = os.path.expanduser('~/.openclaw/openclaw.json')
OPENCLAW_AUTH = os.path.expanduser('~/.openclaw/agents/main/agent/auth-profiles.json')
MAX_TOKENS = 8192


def load_openclaw_api_config():
    """
    Return (api_url, access_token, model) for LLM calls.

    Priority:
      1. Env vars: LLM_API_URL, LLM_API_KEY, LLM_MODEL
      2. OpenClaw auth-profiles.json (minimax-portal:default access token)
    """
    env_url = os.environ.get('LLM_API_URL')
    env_key = os.environ.get('LLM_API_KEY')
    env_model = os.environ.get('LLM_MODEL')

    if env_url and env_key:
        url = env_url.rstrip('/')
        if not url.endswith('/messages'):
            url += '/v1/messages'
        return url, env_key, env_model or 'MiniMax-M2.7'

    # Read model from openclaw.json
    model_id = 'MiniMax-M2.7'
    base_url = 'https://api.minimax.io/anthropic'
    try:
        with open(OPENCLAW_CONFIG) as f:
            cfg = json.load(f)
        raw_model = (cfg.get('agents', {}).get('defaults', {})
                        .get('model', {}).get('primary', model_id))
        if '/' in raw_model:
            model_id = raw_model.split('/', 1)[1]
        # Get the provider's baseUrl
        provider_key = raw_model.split('/')[0] if '/' in raw_model else 'minimax-portal'
        providers = cfg.get('models', {}).get('providers', {})
        if provider_key in providers:
            base_url = providers[provider_key].get('baseUrl', base_url)
    except Exception:
        pass

    api_url = base_url.rstrip('/') + '/v1/messages'

    # Read access token from auth-profiles.json
    try:
        with open(OPENCLAW_AUTH) as f:
            auth = json.load(f)
        profiles = auth.get('profiles', {})
        # Find minimax-portal profile
        for key, profile in profiles.items():
            if 'minimax-portal' in key and profile.get('type') == 'oauth':
                token = profile.get('access', '')
                if token:
                    return api_url, token, env_model or model_id
    except Exception as e:
        pass

    raise SystemExit(
        f'Cannot find Minimax access token in {OPENCLAW_AUTH}\n'
        'Set LLM_API_URL and LLM_API_KEY env vars to override.'
    )

TRANSLATION_PROMPT = '''You are a Pine Script to Python converter for backtesting.

Convert the ENTRY LOGIC ONLY from this Pine Script strategy into a Python function called `generate_entries`.

EXACT FUNCTION SIGNATURE REQUIRED:
```python
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
```

RULES:
1. Only replicate strategy.entry() calls. IGNORE strategy.exit(), strategy.close(), stop losses.
2. Use ONLY pandas and numpy (no other libraries needed).
3. Pine ta.ema(src, len) = src.ewm(span=len, adjust=False).mean()
4. Pine ta.sma(src, len) = src.rolling(len).mean()
5. Pine ta.rsi(src, len): implement Wilder RSI manually
6. Pine ta.atr(len): implement Wilder ATR manually
7. Pine ta.crossover(a, b) at bar i: a.iloc[i] > b.iloc[i] and a.iloc[i-1] <= b.iloc[i-1]
8. Pine ta.crossunder(a, b) at bar i: a.iloc[i] < b.iloc[i] and a.iloc[i-1] >= b.iloc[i-1]
9. Pine source[1] = previous bar = use shift(1) or .iloc[i-1]
10. Entry price = df['close'].iloc[i] unless strategy uses specific price
11. Set trade_num starting at 1, increment for each entry
12. Set entry_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
13. Set raw_price_a = raw_price_b = entry_price_guess
14. Build a boolean Series for each condition, then iterate with a for loop
15. Skip bars where required indicators are NaN
16. If the strategy only has long entries, only emit long entries. Same for short.
17. If strategy has both long and short, emit both directions.

PINE SCRIPT SOURCE TO TRANSLATE:
```pine
{pine_source}
```

Output ONLY the Python function body (starting with `import pandas as pd`), no explanation, no markdown fences.
The output must be valid Python that can be exec()'d directly.'''


def slugify_id(strategy_id):
    s = str(strategy_id).replace(':', '_').replace(';', '_').replace('/', '_')
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s)[:180]


def call_llm_api(pine_source, api_url, api_key, model):
    """
    Call LLM via Anthropic-compatible API with the translation prompt.
    Returns (success, code_or_error).
    """
    prompt = TRANSLATION_PROMPT.replace('{pine_source}', pine_source[:12000])

    headers = {
        'Authorization': f'Bearer {api_key}',
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
    }
    body = {
        'model': model,
        'max_tokens': MAX_TOKENS,
        'messages': [{'role': 'user', 'content': prompt}],
    }

    resp = requests.post(api_url, headers=headers, json=body, timeout=(600, 600))

    if resp.status_code != 200:
        return False, f'API error {resp.status_code}: {resp.text[:500]}'

    data = resp.json()
    content = data.get('content', [])

    # First pass: prefer explicit text blocks
    for block in content:
        if block.get('type') == 'text':
            text = block['text'].strip()
            text = re.sub(r'^```[a-z]*\n?', '', text, flags=re.MULTILINE)
            text = re.sub(r'\n?```$', '', text, flags=re.MULTILINE)
            return True, text.strip()

    # Fallback: extract Python code embedded in a thinking block
    # (happens when the model runs out of tokens mid-think)
    for block in content:
        thinking_text = block.get('thinking') or (block.get('text') if block.get('type') == 'thinking' else None)
        if thinking_text:
            # Look for Python code block inside the thinking
            match = re.search(r'```python\n(.*?)```', thinking_text, re.DOTALL)
            if match:
                return True, match.group(1).strip()

    return False, f'No text content in response: {str(data)[:300]}'


def validate_python(code):
    """Validate Python syntax and that generate_entries function exists."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f'SyntaxError: {e}'

    # Check for generate_entries function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'generate_entries':
            return True, None

    return False, 'generate_entries function not found in output'


def translate_one(slug, pine_path, api_url, api_key, model, out_path):
    """Translate a single Pine script. Returns (success, info_dict)."""
    with open(pine_path, encoding='utf-8') as f:
        pine_source = f.read()

    if len(pine_source.strip()) < 50:
        return False, {'error': 'Pine source too short to be a real strategy'}

    ok, result = call_llm_api(pine_source, api_url, api_key, model)
    if not ok:
        return False, {'error': result}

    valid, err = validate_python(result)
    if not valid:
        # Save the bad output for debugging
        debug_path = out_path + '.invalid'
        with open(debug_path, 'w') as f:
            f.write(result)
        return False, {'error': f'Invalid Python: {err}', 'debug_path': debug_path}

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(result)

    return True, {'chars': len(result)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--slug', default=None, help='Single strategy ID slug to translate')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--force', action='store_true', help='Re-translate even if .py exists')
    p.add_argument('--dry-run', action='store_true', help='Test API connection without saving')
    return p.parse_args()


def main():
    args = parse_args()

    api_url, api_key, model = load_openclaw_api_config()
    print(f'[translate] API: {api_url}  model: {model}')

    if args.dry_run:
        print('[translate] Dry run: testing API connection...')
        ok, result = call_llm_api('strategy("test", overlay=true)\nstrategy.entry("L", strategy.long, when=close > open)', api_url, api_key, model)
        if ok:
            print(f'[translate] API OK. Response snippet: {result[:200]}')
        else:
            print(f'[translate] API FAILED: {result}')
        return

    os.makedirs(TRANSLATED_DIR, exist_ok=True)

    # Load classified to get id→name mapping
    with open(CLASSIFIED_FILE) as f:
        classified = json.load(f)

    id_to_name = {s.get('id', ''): s['name'] for s in classified}

    # Find all .pine files
    if args.slug:
        pine_files = [f'{args.slug}.pine']
    else:
        pine_files = [f for f in os.listdir(PINE_DIR) if f.endswith('.pine')]
        pine_files.sort()

    if args.limit:
        pine_files = pine_files[:args.limit]

    total = len(pine_files)
    print(f'[translate] {total} Pine scripts to translate using {model}')

    # Load existing log
    log = {}
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            log = json.load(f)

    translated = 0
    skipped = 0
    errored = 0

    for i, pine_file in enumerate(pine_files):
        slug = pine_file[:-5]  # remove .pine
        pine_path = os.path.join(PINE_DIR, pine_file)
        out_path = os.path.join(TRANSLATED_DIR, f'{slug}.py')
        name = id_to_name.get(slug.replace('_', ':').replace('USER_', 'USER;'), slug)

        label = f'[{i+1}/{total}] {slug[:60]}'

        if not args.force and os.path.exists(out_path):
            print(f'{label} → skip (exists)')
            skipped += 1
            continue

        print(f'{label} → translating...', end=' ', flush=True)

        ok, info = translate_one(slug, pine_path, api_url, api_key, model, out_path)

        if ok:
            print(f'OK ({info["chars"]} chars)')
            log[slug] = {'status': 'ok', 'chars': info['chars']}
            translated += 1
        else:
            print(f'FAILED: {info["error"][:100]}')
            log[slug] = {'status': 'failed', 'error': info['error']}
            errored += 1

        # Rate limit: ~1 req/sec to avoid hitting Anthropic limits
        time.sleep(1.0)

        # Save log every 20 translations
        if (i + 1) % 20 == 0:
            with open(LOG_FILE, 'w') as f:
                json.dump(log, f, indent=2)

    with open(LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)

    print(f'\n[translate] Done. translated={translated} skipped={skipped} failed={errored}')
    print(f'[translate] Translated Python files in: {TRANSLATED_DIR}')
    print(f'[translate] Translation log in: {LOG_FILE}')


if __name__ == '__main__':
    main()
