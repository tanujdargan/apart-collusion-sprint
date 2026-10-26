"""Hard-cap spend guard for the verification extension (cap $8.00 TOTAL new spend; notes/verify_plan.md §7).

  .venv-analysis/bin/python exp/verify_guard.py baseline
  .venv-analysis/bin/python exp/verify_guard.py run <stage> <n_containers> <wall_limit_s> <L40S|H200> -- <modal run args...>
  .venv-analysis/bin/python exp/verify_guard.py status

Accounting (data/verify/spend_ledger.json):
- billed delta = Modal month-to-date (all workspace apps; conservative) - baseline. Billing lags, so it is never trusted alone.
- per stage, a conservative COMPUTED upper bound = n_containers x (measured local wall + 60 s teardown/idle slack) x GPU
  rate (+ 10% for CPU/memory). Containers cannot outlive the local client: on wall-limit expiry the client is
  interrupted and the app stopped, and the Modal per-input timeout + max_containers are set to the same limits.
- spent = max(billed delta, sum of computed upper bounds of finished stages).
- launch allowed only if spent + n_containers x (wall_limit + 60) x rate x 1.1 <= cap (the stage's worst case).
"""
import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import os
LEDGER = ROOT / os.environ.get("VERIFY_LEDGER", "data/verify/spend_ledger.json")  # v2: data/verify/spend_ledger_v2.json
CAP = float(os.environ.get("VERIFY_CAP", "8.00"))
RATE = {"L40S": 0.000542, "H200": 0.001261}
SLACK_S, OVERHEAD = 60, 1.10
MODAL = str(ROOT / ".venv/bin/modal")


def month_total():
    out = subprocess.run([MODAL, "billing", "report", "--for", "this month", "--json"], capture_output=True, text=True,
                         timeout=120)
    return round(sum(float(r["cost"]) for r in json.loads(out.stdout)), 4)


def running_apps():
    out = subprocess.run([MODAL, "app", "list", "--json"], capture_output=True, text=True, timeout=120)
    return [a for a in json.loads(out.stdout or "[]") if str(a.get("State", a.get("state", ""))).lower() not in ("stopped",)]


def load():
    return json.load(open(LEDGER))


def save(L):
    json.dump(L, open(LEDGER, "w"), indent=1)


def spent(L):
    billed = month_total() - L["baseline_month_total"]
    computed = sum(s["computed_upper"] for s in L["stages"])
    return max(billed, computed), billed, computed


def bound(n, wall, gpu):
    return n * (wall + SLACK_S) * RATE[gpu] * OVERHEAD


def main():
    cmd = sys.argv[1]
    if cmd == "baseline":
        assert not LEDGER.exists(), "baseline already recorded"
        apps = running_apps()
        assert not apps, f"apps running before baseline: {apps}"
        b = month_total()
        save(dict(cap=CAP, baseline_month_total=b, baseline_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  prior_reference=float(os.environ.get("VERIFY_PRIOR_REF", "42.448")), note="baseline == Phase 0 month-to-date (no spend in between) if equal to "
                  "prior_reference; earlier spend is excluded from the cap", stages=[]))
        print("baseline", b)
    elif cmd == "status":
        L = load()
        s, b, c = spent(L)
        print(f"spent(max) ${s:.3f}  billed ${b:.3f}  computed ${c:.3f}  cap ${CAP}  remaining ${CAP - s:.3f}")
        for st in L["stages"]:
            print(" ", st["stage"], st["status"], f"wall {st['wall_s']:.0f}s", f"upper ${st['computed_upper']:.3f}")
    elif cmd == "run":
        stage, n, wl, gpu = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
        args = sys.argv[sys.argv.index("--") + 1:]
        L = load()
        s, b, c = spent(L)
        worst = bound(n, wl, gpu)
        if s + worst > CAP:
            print(f"REFUSED {stage}: spent ${s:.3f} + worst case ${worst:.3f} > cap ${CAP}")
            sys.exit(2)
        print(f"launch {stage}: spent ${s:.3f} (billed {b:.3f}, computed {c:.3f}) + worst ${worst:.3f} <= ${CAP}", flush=True)
        t0 = time.time()
        log = open(ROOT / f"data/verify/logs/{stage}.log", "w")
        p = subprocess.Popen([MODAL, "run"] + args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        app_id, status = None, "ok"
        try:
            import selectors
            sel = selectors.DefaultSelector()
            sel.register(p.stdout, selectors.EVENT_READ)
            while True:
                if time.time() - t0 > wl:
                    status = "wall_limit"
                    break
                if not sel.select(timeout=1):
                    if p.poll() is not None:
                        break
                    continue
                line = p.stdout.readline()
                if not line:          # EOF: wait for exit
                    p.wait(60)
                    break
                if True:
                    log.write(line); log.flush()
                    sys.stdout.write(line); sys.stdout.flush()
                    m = re.search(r"(ap-[A-Za-z0-9]+)", line)
                    if m and not app_id:
                        app_id = m.group(1)
        finally:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
                try:
                    p.wait(30)
                except subprocess.TimeoutExpired:
                    p.kill()
            if app_id and status != "ok":
                subprocess.run([MODAL, "app", "stop", app_id], capture_output=True)
            if p.returncode not in (0, None) and status == "ok":
                status = f"exit_{p.returncode}"
            wall = time.time() - t0
            up = bound(n, wall, gpu)
            L = load()
            L["stages"].append(dict(stage=stage, n=n, gpu=gpu, wall_limit=wl, wall_s=wall, computed_upper=up,
                                    worst_reserved=worst, status=status, app_id=app_id,
                                    end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            save(L)
            time.sleep(5)
            left = running_apps()
            print(f"[guard] {stage} {status} wall {wall:.0f}s computed upper ${up:.3f}; running apps after: {len(left)}")
            if left and app_id:
                subprocess.run([MODAL, "app", "stop", app_id], capture_output=True)
        sys.exit(0 if status == "ok" else 1)


if __name__ == "__main__":
    main()
