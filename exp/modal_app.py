"""Cut the Wire — Modal GPU pipeline (generation, receiver decode, activation extraction, interventions).

Usage (from project root):
  modal run exp/modal_app.py::gate                       # Stage A gate (5 seeds -> 90 episodes)
  modal run exp/modal_app.py::stage_b                    # Stage B (17 seeds -> 306 episodes) + activations
  modal run exp/modal_app.py::intervene --direction-file X.npy --mode ablate --layers all --tag foo
Other models: add --model <hf id> --out-tag <short_tag> to any entrypoint. Volume paths become /<short_tag>/{data,acts,
interv}/... and local outputs go to data_models/<short_tag>/ (mirrors data/). The default (no --out-tag) paths are the
original 7B ones. Models that need a newer transformers (e.g. OLMo-2): set CUTWIRE_TRANSFORMERS=4.48.3 in the shell.
Layer convention: layer l (0..L-1) = output of decoder layer l (== HF hidden_states[l+1], but captured with a
forward hook so the last layer is pre-final-norm).
"""
import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TRACE_LAYERS = [8, 12, 16, 20, 24]
# condition -> (receiver is colluding?, which sender message)
COND = {"C1": (True, "collude_msg"), "C2": (True, "honest_msg"), "C3": (False, "collude_msg"),
        "C4": (True, "offtopic_msg"), "C5": (True, "collude_msg"), "C6": ("explicit", "collude_msg")}
# C6 = no covert channel: honest receiver told the state openly (public indicator), same DECODE/BET format and bet
# rule. Tests whether a subspace is collusion-specific or just the receiver's answer/bet-tier representation.
# C5 = scheme mismatch: C1's message, but the receiver's private transcript uses env.alt_scheme_id(ep), a scheme
# that does not contain the message's topic. Implemented by swapping the episode's transcript (env.mismatch).


def _ep(r, cond):
    import env
    return env.mismatch(r) if cond == "C5" else r
GEN = dict(sender_temperature=0.7, sender_top_p=1.0, sender_max_new=120, decode_max_new=24, recall_max_new=6,
           arc_max_new=4)

# transformers version is baked into the image; override via env var for model families it does not support (the env
# var is also set inside the image so the container re-derives the same image definition).
TF_VERSION = os.environ.get("CUTWIRE_TRANSFORMERS", "4.46.3")
vol = modal.Volume.from_name("cutwire-vol", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.5.1", f"transformers=={TF_VERSION}", "accelerate==1.1.1", "datasets==3.1.0",
                      "numpy==2.1.3", "hf_transfer==0.1.8")
         .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
               **({"CUTWIRE_TRANSFORMERS": TF_VERSION} if TF_VERSION != "4.46.3" else {})})
         .add_local_file(HERE / "env.py", "/root/env.py"))
app = modal.App("cutwire", image=image)


def _vp(out_tag):
    """Volume path prefix for a model namespace ('' = original 7B layout)."""
    return f"/{out_tag}" if out_tag else ""


def _local(out_tag):
    return ROOT / (f"data_models/{out_tag}" if out_tag else "data")


@app.cls(gpu="L40S", volumes={"/vol": vol}, timeout=3600, scaledown_window=30)
class Runner:
    model_id: str = modal.parameter(default=DEFAULT_MODEL)

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16,
                                                          device_map="cuda", attn_implementation="sdpa").eval()
        self.layers = self.model.model.layers
        self.L, self.D = len(self.layers), self.model.config.hidden_size
        self.eot = self._end_of_turn()
        # activation-extraction batch size: 16 for <= 7B-sized residual streams, smaller for bigger models
        self.acts_bs = 16 if self.L * self.D <= 28 * 3584 else 8
        vol.commit()
        print(f"loaded {self.model_id} L={self.L} D={self.D} eot={self.eot!r} acts_bs={self.acts_bs} "
              f"in {time.time() - t0:.0f}s")

    def _end_of_turn(self):
        """End-of-turn string the chat template appends after an assistant message (Qwen: '<|im_end|>')."""
        probe = [dict(role="user", content="hi")]
        a = self.tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
        b = self.tok.apply_chat_template(probe + [dict(role="assistant", content="XQZW")], tokenize=False)
        suffix = b.split("XQZW", 1)[1] if "XQZW" in b else ""
        eot = suffix.split()[0] if suffix.strip() else self.tok.eos_token
        print("chat template gen-prompt tail:", repr(a[-40:]), "| assistant suffix:", repr(suffix), "-> eot", repr(eot))
        return eot

    # ------------------------------------------------------------------ helpers
    def _tmpl(self, msgs):
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _generate(self, prompts, max_new, temperature=None, top_p=1.0, seed=0, bs=48):
        import torch
        self.tok.padding_side = "left"
        outs = []
        for i in range(0, len(prompts), bs):
            enc = self.tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            kw = dict(do_sample=True, temperature=temperature, top_p=top_p) if temperature else dict(do_sample=False)
            torch.manual_seed(seed * 100003 + i)
            with torch.no_grad():
                g = self.model.generate(**enc, max_new_tokens=max_new, pad_token_id=self.tok.eos_token_id,
                                        top_k=None, repetition_penalty=1.0, **({} if temperature else dict(temperature=None, top_p=None)),
                                        **kw)
            outs += self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return outs

    def _span(self, text, prefix, msg):
        """Token [start, end) of the sender message inside templated text (last occurrence of prefix+msg)."""
        c0 = text.rfind(prefix + msg)
        assert c0 >= 0, "message not found in prompt"
        cs, ce = c0 + len(prefix), c0 + len(prefix) + len(msg)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        idx = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > cs and a < ce]
        return idx[0], idx[-1] + 1, len(enc["input_ids"])

    def _acts(self, texts, spans, plens, trace_idx=(), bs=None):
        """One forward over prompt+response+<|im_end|> (causal, so prompt positions equal a prompt-only pass).
        Returns fp16 [N,L,D] arrays: mean over sender span, last prompt token (decision position), last token of
        full transcript (end-of-turn, Rose et al. pooling); plus traces dict."""
        import numpy as np
        import torch
        self.tok.padding_side = "right"
        bs = bs or self.acts_bs
        N = len(texts)
        mean = np.zeros((N, self.L, self.D), np.float16)
        last = np.zeros((N, self.L, self.D), np.float16)
        full = np.zeros((N, self.L, self.D), np.float16)
        traces = {}
        cap = {}
        hooks = [lay.register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(l, o[0] if isinstance(o, tuple) else o))
                 for l, lay in enumerate(self.layers)]
        try:
            for b0 in range(0, N, bs):
                enc = self.tok(texts[b0:b0 + bs], return_tensors="pt", padding=True,
                               add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    self.model.model(**enc)
                H = torch.stack([cap[l] for l in range(self.L)], 1)  # [B,L,T,D]
                lens = enc["attention_mask"].sum(1)
                for j in range(H.shape[0]):
                    n = b0 + j
                    s, e, _ = spans[n]
                    mean[n] = H[j, :, s:e].float().mean(1).cpu().numpy()
                    last[n] = H[j, :, plens[n] - 1].float().cpu().numpy()
                    full[n] = H[j, :, lens[j] - 1].float().cpu().numpy()
                    if n in trace_idx:
                        ee = min(e + 10, plens[n])
                        traces[n] = H[j, TRACE_LAYERS, s:ee].float().cpu().numpy().astype(np.float16)
                cap.clear()
        finally:
            for h in hooks:
                h.remove()
        return mean, last, full, traces

    def _install(self, direction, mode, layers, alpha, subspace=False):
        """Intervention hooks on decoder-layer outputs (and embedding output if layers=='all' and mode=='ablate').
        direction (RAW residual space): [D] one direction for all layers; [L,D] per-layer direction; [k,D] k-dim
        subspace for all layers (2-D arrays are read as [k,D] if subspace=True or shape[0]!=L); [L,k,D] per-layer
        subspace. Ablate: h <- h - (h @ Q) Q^T with Q = orthonormal basis (QR) of the layer's vectors.
        Steer: h <- h + alpha * u, u = first vector of the layer, unit-normalised (1-D only)."""
        import torch
        d = torch.tensor(direction, dtype=torch.float32, device="cuda")
        if d.ndim == 1:
            d = d[None, None]                      # [1,1,D]
        elif d.ndim == 2:
            d = d[None] if (subspace or d.shape[0] != self.L) else d[:, None]  # [1,k,D] or [L,1,D]
        assert d.ndim == 3 and d.shape[-1] == self.D and d.shape[0] in (1, self.L), tuple(d.shape)
        Qs = [torch.linalg.qr(v.T)[0] for v in d]  # each [D,k], orthonormal columns
        us = [v[0] / v[0].norm() for v in d]
        lidx = list(range(self.L)) if layers in (None, "all") else [int(x) for x in layers]
        pick = (lambda l: 0) if d.shape[0] == 1 else (lambda l: l)

        def fn(h, l):
            hf = h.float()
            if mode == "ablate":
                Q = Qs[pick(l)]
                hf = hf - (hf @ Q) @ Q.T
            elif mode == "steer":
                hf = hf + alpha * us[pick(l)]
            return hf.to(h.dtype)

        def hook(m, i, o, l):
            if isinstance(o, tuple):
                return (fn(o[0], l),) + o[1:]
            return fn(o, l)
        hs = [self.layers[l].register_forward_hook(lambda m, i, o, l=l: hook(m, i, o, l)) for l in lidx]
        if mode == "ablate" and layers in (None, "all"):
            hs.append(self.model.model.embed_tokens.register_forward_hook(lambda m, i, o: fn(o, 0)))
        return hs

    # ------------------------------------------------------------------ receivers
    def _receivers(self, eps, msgs, cond):
        """C1 colluding rx + colluding msg; C2 colluding rx + honest msg; C3 honest rx + colluding msg;
        C4 colluding rx + honest msg that mentions an off-scheme topic; C5 C1 msg under a mismatched scheme."""
        import env
        colluding_rx, mkey = COND[cond]
        texts = [self._tmpl(env.receiver_msgs(_ep(e, cond), msgs[e["id"]][mkey], colluding_rx)) for e in eps]
        outs = self._generate(texts, GEN["decode_max_new"])
        rows = []
        for e, t, o in zip(eps, texts, outs):
            st, bet = env.parse_decode(o) if colluding_rx else (None, env.parse_bet(o))
            rows.append(dict(id=e["id"], scheme_id=e["scheme_id"], state=e["state"], condition=cond,
                             msg=msgs[e["id"]][mkey], output=o, decoded=st, bet=bet,
                             **({"alt_scheme_id": env.alt_scheme_id(e)} if cond == "C5" else {})))
        return rows, texts

    def _recall(self, eps, msgs):
        import env
        texts = [self._tmpl(env.receiver_msgs(e, msgs[e["id"]]["collude_msg"], True, "recall")) for e in eps]
        outs = self._generate(texts, GEN["recall_max_new"])
        rows = [dict(id=e["id"], topic=e["topic"], output=o, answer=env.parse_recall(o, e["recall_options"]))
                for e, o in zip(eps, outs)]
        return rows, sum(r["answer"] == r["topic"] for r in rows) / len(rows)

    def _arc(self, n=100):
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").select(range(n))
        texts, golds, labs = [], [], []
        for x in ds:
            ch = "\n".join(f"{l}. {t}" for l, t in zip(x["choices"]["label"], x["choices"]["text"]))
            texts.append(self._tmpl([dict(role="user", content=f"Question: {x['question']}\n{ch}\n"
                                      "Answer with the letter of the correct choice only.")]))
            golds.append(x["answerKey"])
            labs.append(x["choices"]["label"])
        outs = self._generate(texts, GEN["arc_max_new"])
        pred = [next((c for c in o.strip().replace(".", " ").split()[:1] if c in lab), None) if o.strip() else None
                for o, lab in zip(outs, labs)]
        return sum(p == g for p, g in zip(pred, golds)) / n

    # ------------------------------------------------------------------ public methods
    @modal.method()
    def gen_messages(self, seeds, gen_seed=0):
        import env
        eps = env.episodes(seeds)
        res = {}
        for gi, (colluding, key) in enumerate(((True, "collude"), (False, "honest"), ("offtopic", "offtopic"))):
            texts = [self._tmpl(env.sender_msgs(e, colluding)) for e in eps]
            outs = self._generate(texts, GEN["sender_max_new"], temperature=GEN["sender_temperature"],
                                  top_p=GEN["sender_top_p"], seed=gen_seed + gi)
            for e, o in zip(eps, outs):
                r = res.setdefault(e["id"], dict(e))
                r[key + "_raw"] = o
                r[key + "_msg"] = env.parse_talk(o) or "Good luck everyone."
        for r in res.values():
            r["collude_clean"] = env.sender_check(r, r["collude_msg"])["clean_target"]
        return list(res.values())

    @modal.method()
    def run_receivers(self, msg_rows, conditions=("C1", "C2", "C3", "C4"), recall=True):
        msgs = {r["id"]: r for r in msg_rows}
        out = {c: self._receivers(msg_rows, msgs, c)[0] for c in conditions}
        if recall:
            out["recall"], out["recall_acc"] = self._recall(msg_rows, msgs)
        return out

    @modal.method()
    def extract(self, msg_rows, outputs, conditions=("C1", "C2", "C3", "C4"), n_trace=40, out_dir="/vol/acts"):
        return self._extract(msg_rows, outputs, conditions, n_trace, out_dir)

    def _extract(self, msg_rows, outputs, conditions, n_trace, out_dir):
        """outputs: {cond: {episode_id: generated receiver text}} (used for the full-transcript pooling)."""
        import numpy as np
        import env
        os.makedirs(out_dir, exist_ok=True)
        meta = dict(model=self.model_id, n_layers=self.L, d_model=self.D, trace_layers=TRACE_LAYERS,
                    layer_convention="layer l = output of decoder layer l (forward hook, pre-final-norm)",
                    ids=[r["id"] for r in msg_rows], scheme_id=[r["scheme_id"] for r in msg_rows],
                    state=[r["state"] for r in msg_rows], conditions={})
        # trace examples: evenly spread over episodes so all schemes/states appear
        trace_idx = set(np.linspace(0, len(msg_rows) - 1, n_trace).round().astype(int).tolist())
        for c in conditions:
            colluding_rx, mkey = COND[c]
            texts, fulls, spans, plens, bad = [], [], [], [], 0
            for r in msg_rows:
                t = self._tmpl(env.receiver_msgs(_ep(r, c), r[mkey], colluding_rx))
                f = t + outputs[c][r["id"]] + self.eot
                sp = self._span(t, f"\n{r['sender']}: ", r[mkey])
                fid = self.tok(f, add_special_tokens=False)["input_ids"]
                pid = self.tok(t, add_special_tokens=False)["input_ids"]
                bad += fid[:len(pid)] != pid
                texts.append(t), fulls.append(f), spans.append(sp), plens.append(sp[2])
            assert bad == 0, f"{bad} prompts are not a token-prefix of their full transcript"
            mean, last, full, traces = self._acts(fulls, spans, plens, trace_idx if c in ("C1", "C2", "C5") else ())
            np.savez(f"{out_dir}/{c}.npz", mean=mean, last=last, full_last=full)
            norms = np.linalg.norm(mean.astype(np.float32), axis=-1).mean(0).round(2).tolist()
            lnorms = np.linalg.norm(last.astype(np.float32), axis=-1).mean(0).round(2).tolist()
            cm = dict(sender_msg=[r[mkey] for r in msg_rows], receiver_output=[outputs[c][r["id"]] for r in msg_rows],
                      spans=spans, mean_span_norm_per_layer=norms, last_token_norm_per_layer=lnorms)
            if traces:
                ks = sorted(traces)
                T = max(traces[k].shape[1] for k in ks)
                arr = np.zeros((len(ks), len(TRACE_LAYERS), T, self.D), np.float16)
                for i, k in enumerate(ks):
                    arr[i, :, :traces[k].shape[1]] = traces[k]
                np.savez(f"{out_dir}/{c}_traces.npz", acts=arr, lengths=np.array([traces[k].shape[1] for k in ks]),
                         idx=np.array(ks))
                cm["traces"] = dict(idx=ks, tokens={str(k): [self.tok.decode([t]) for t in self.tok(
                    texts[k], add_special_tokens=False)["input_ids"][spans[k][0]:min(spans[k][1] + 10, spans[k][2])]]
                    for k in ks}, msg_len={str(k): spans[k][1] - spans[k][0] for k in ks})
            if c == "C5":
                cm["alt_scheme_id"] = [env.alt_scheme_id(r) for r in msg_rows]
            meta["conditions"][c] = cm
            print(c, mean.shape, "traces", len(traces))
        json.dump(meta, open(f"{out_dir}/meta.json", "w"))
        vol.commit()
        return meta

    @modal.method()
    def intervene(self, direction, mode="ablate", layers="all", alpha=0.0, conditions=("C1",), n=None,
                  tag="run", do_arc=True, do_acts=True, subspace=False, out_tag=""):
        """Receiver-side intervention. direction: RAW residual-space direction, array [D] or [L,D] (unit-normalised
        here; no standardisation applied), 'random', or None for mode='none' baseline."""
        import numpy as np
        import env
        vp = _vp(out_tag)
        rows = [json.loads(l) for l in open(f"/vol{vp}/data/messages.jsonl")]
        rows = rows if n is None else [rows[i] for i in np.linspace(0, len(rows) - 1, n).round().astype(int)]
        msgs = {r["id"]: r for r in rows}
        if isinstance(direction, str) and direction == "random":
            direction = np.random.default_rng(0).standard_normal(self.D).astype(np.float32)
        hs = [] if mode == "none" else self._install(np.asarray(direction), mode, layers, alpha, subspace)
        t0 = time.time()
        res = dict(tag=tag, mode=mode, layers=layers, alpha=alpha, n=len(rows), model=self.model_id)
        try:
            outputs = {}
            for c in conditions:
                rr, _ = self._receivers(rows, msgs, c)
                outputs[c] = {x["id"]: x["output"] for x in rr}
                res[c] = env.decode_metrics(rr) if c != "C3" else env.decode_metrics(rr) | dict(decode_acc=None)
                res[c]["decoded_dist"] = {str(k): sum(x["decoded"] == k for x in rr) for k in env.STATES + [None]}
                res[c + "_rows"] = rr
            res["recall"], res["recall_acc"] = self._recall(rows, msgs)
            if do_arc:
                res["arc_acc"] = self._arc(100)
            if do_acts:
                od = f"/vol{vp}/interv/{tag}"
                # hooks stay installed: activations are captured after the intervention is applied
                self._extract(rows, outputs, tuple(conditions), 0, od)
                res["acts_dir"] = od
        finally:
            for h in hs:
                h.remove()
        res["seconds"] = time.time() - t0
        os.makedirs(f"/vol{vp}/interv/{tag}", exist_ok=True)
        json.dump(res, open(f"/vol{vp}/interv/{tag}/metrics.json", "w"))
        vol.commit()
        return res


# ---------------------------------------------------------------------- local entrypoints
def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@app.local_entrypoint()
def gate(model: str = DEFAULT_MODEL, seeds: int = 5, tag: str = "gate", out_tag: str = ""):
    t0 = time.time()
    r = Runner(model_id=model)
    rows = r.gen_messages.remote(list(range(seeds)), gen_seed=0)
    out = r.run_receivers.remote(rows)
    _write_jsonl(_local(out_tag) / f"{tag}_messages.jsonl", rows)
    _write_jsonl(_local(out_tag) / f"{tag}_receivers.jsonl", sum((out[c] for c in COND if c in out), []) + out["recall"])
    print(f"wall {time.time() - t0:.0f}s; analyse with: python exp/analyze_gate.py {tag}")


@app.local_entrypoint()
def stage_b(model: str = DEFAULT_MODEL, seeds: int = 17, out_tag: str = "", conditions: str = "C1,C2,C3,C4"):
    """out_tag: namespace for non-default models (volume /<out_tag>/..., local data_models/<out_tag>/).
    conditions: receiver conditions to run and extract (new models use C1,C2,C3,C4,C5 in one pass)."""
    t0 = time.time()
    r = Runner(model_id=model)
    conds = tuple(conditions.split(","))
    loc, vp = _local(out_tag), _vp(out_tag)
    rows = r.gen_messages.remote(list(range(seeds)), gen_seed=1000)
    _write_jsonl(loc / "messages.jsonl", rows)
    with vol.batch_upload(force=True) as b:
        b.put_file(str(loc / "messages.jsonl"), f"{vp}/data/messages.jsonl")
    out = r.run_receivers.remote(rows, conditions=conds)
    _write_jsonl(loc / "receiver_outputs.jsonl", sum((out[c] for c in COND if c in out), []))
    _write_jsonl(loc / "recall_outputs.jsonl", out["recall"])
    print("recall_acc", out["recall_acc"])
    outputs = {c: {x["id"]: x["output"] for x in out[c]} for c in COND if c in out}
    meta = r.extract.remote(rows, outputs, conditions=conds, out_dir=f"/vol{vp}/acts")
    print("meta conditions", list(meta["conditions"]), f"wall {time.time() - t0:.0f}s")


@app.local_entrypoint()
def intervene(direction_file: str = "", mode: str = "ablate", layers: str = "all", alpha: float = 0.0,
              conditions: str = "C1,C2,C3,C4", n: int = 0, tag: str = "run", model: str = DEFAULT_MODEL,
              arc: bool = True, acts: bool = True, subspace: bool = False, pull: bool = True, out_tag: str = ""):
    """direction_file: local .npy [D] or [L,D]; 'random' -> random unit vector (seed 0); '' with mode=none."""
    import numpy as np
    if direction_file == "random":
        d = "random"
    elif direction_file:
        d = np.load(direction_file).astype(np.float32)
    else:
        d = None
    lay = "all" if layers == "all" else [int(x) for x in layers.split(",")]
    res = Runner(model_id=model).intervene.remote(d, mode, lay, alpha, tuple(conditions.split(",")), n or None, tag,
                                                  arc, acts, subspace, out_tag)
    od = _local(out_tag) / f"interv/{tag}"
    vp = _vp(out_tag)
    od.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(od / "metrics.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("_rows") and k != "recall"}, indent=1))
    if acts and pull:
        import subprocess
        for c in conditions.split(","):
            subprocess.run(["modal", "volume", "get", "--force", "cutwire-vol", f"{vp}/interv/{tag}/{c}.npz", str(od)],
                           check=False)
        subprocess.run(["modal", "volume", "get", "--force", "cutwire-vol", f"{vp}/interv/{tag}/meta.json", str(od)],
                       check=False)


@app.local_entrypoint()
def stage_c5(model: str = DEFAULT_MODEL):
    """Add condition C5 to existing Stage B data without touching C1-C4 files; merges meta into data/acts/meta.json."""
    import subprocess
    rows = [json.loads(l) for l in open(ROOT / "data/messages.jsonl")]
    r = Runner(model_id=model)
    out = r.run_receivers.remote(rows, conditions=("C5",), recall=False)
    _write_jsonl(ROOT / "data/receiver_outputs_C5.jsonl", out["C5"])
    meta5 = r.extract.remote(rows, {"C5": {x["id"]: x["output"] for x in out["C5"]}}, conditions=("C5",),
                             out_dir="/vol/acts_c5")
    for f in ("C5.npz", "C5_traces.npz"):
        subprocess.run(["modal", "volume", "get", "--force", "cutwire-vol", f"/acts_c5/{f}", str(ROOT / "data/acts")],
                       check=True)
    mp = ROOT / "data/acts/meta.json"
    meta = json.load(open(mp))
    assert meta["ids"] == meta5["ids"], "row order mismatch"
    meta["conditions"]["C5"] = meta5["conditions"]["C5"]
    json.dump(meta, open(mp, "w"))
    print("C5 merged; decoded:", {k: sum(x["decoded"] == k for x in out["C5"]) for k in ("COLD", "NEUTRAL", "HOT", None)})
