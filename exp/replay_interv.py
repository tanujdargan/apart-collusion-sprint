"""Rebuild the exact `modal run ::intervene` command of every committed intervention run from its metrics.json.

  python3 exp/replay_interv.py cmds <ns>    # one shell command per line (Makefile pipes these to xargs)
  python3 exp/replay_interv.py post <ns>    # `probes.py post` commands for runs that have post_probes.json
  <ns> = data | data_models/qwen14b | data_models/phi35mini | data_models/qwen32b

Each run's metrics.json stores tag, mode, layers, alpha, n, model and one key per condition run; --no-arc / --no-acts
are read off the absence of arc_acc / acts_dir. The direction file is the tag's base name in <ns>/directions
(tags: abl_<dir>[_late][_C6], loc_<dir>_{early,late}, steer_state_L<l>_a<alpha>, steer_random_*). smoke_* runs are skipped.
"""
import json
import re
import sys
from pathlib import Path

CONDS = ("C1", "C2", "C3", "C4", "C5", "C6")
TAG = {"qwen14b": "qwen14b", "phi35mini": "phi35mini", "qwen32b": "qwen32b"}


def direction(tag, ns):
    if tag in ("baseline", "base_C6"):
        return ""
    if tag.startswith("steer_random"):
        return f"{ns}/directions/random1d_3.npy"
    if m := re.match(r"steer_state_L(\d+)_a", tag):
        return f"{ns}/directions/state_hotcold_L{m[1]}.npy"
    if tag == "verify_state_global_acts":  # 32B V1 check: global (all-episode) state span
        return f"{ns}/loso/global/state.npy"
    base = re.sub(r"^(abl|loc)_", "", tag)
    base = re.sub(r"(_late)?(_C6)?$", "", base) if tag.startswith("abl_") else re.sub(r"_(early|late)$", "", base)
    return f"{ns}/directions/{base}.npy"


def cmd(p, ns):
    m = json.loads(p.read_text())
    tag, lay = m["tag"], m["layers"]
    a = [".venv/bin/modal run exp/modal_app.py::intervene", f"--tag {tag}", f"--mode {m['mode']}",
         f"--layers {lay if isinstance(lay, str) else ','.join(map(str, lay))}",
         f"--conditions {','.join(c for c in CONDS if c in m)}"]
    if d := direction(tag, ns):
        a.append(f"--direction-file {d}")
    if m["mode"] == "steer":
        a.append(f"--alpha {m['alpha']}")
    if m.get("n", 306) != 306:
        a.append(f"--n {m['n']}")
    if "arc_acc" not in m:
        a.append("--no-arc")
    if "acts_dir" not in m:
        a.append("--no-acts")
    if ns != "data":
        a.append(f"--model {m['model']} --out-tag {TAG[Path(ns).name]}")
        if "32B" in m["model"]:
            a.append("--gpu H200")
    return " ".join(a)


def runs(ns):
    return [p for p in sorted(Path(ns, "interv").glob("*/metrics.json")) if not p.parent.name.startswith("smoke_")]


if __name__ == "__main__":
    what, ns = sys.argv[1], sys.argv[2].rstrip("/")
    for p in runs(ns):
        if what == "cmds":
            print(cmd(p, ns))
        elif (p.parent / "post_probes.json").exists():
            print(f"DATA_DIR={ns} .venv-analysis/bin/python exp/probes.py post {p.parent}")
