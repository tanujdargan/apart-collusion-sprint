#!/bin/bash
# Today's Modal spend (USD) and delta vs the follow-up batch baseline ($7.397). Billing lags a few minutes.
cd "$(dirname "$0")/.." && .venv/bin/modal billing report --for today --json 2>/dev/null | .venv-analysis/bin/python -c "
import json,sys; t=sum(float(r['cost']) for r in json.load(sys.stdin)); print(f'today \${t:.3f}  followups delta \${t-7.397:.3f}')"
