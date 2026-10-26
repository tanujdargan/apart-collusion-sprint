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
            self._pmask = enc["attention_mask"]  # read by the perturbation meter (prefill call only)
            with torch.no_grad():
                g = self.model.generate(**enc, max_new_tokens=max_new, pad_token_id=self.tok.eos_token_id,
                                        top_k=None, repetition_penalty=1.0, **({} if temperature else dict(temperature=None, top_p=None)),
                                        **kw)
            outs += self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        self._pmask = None
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

    def _meter(self, meter, key, h, delta):
        """Accumulate achieved perturbation: sum ||delta||^2, sum ||h||^2 (pre-hook), token count, per phase and location.
        Prefill (T>1) excludes left padding via the generation attention mask; decode steps (T==1) count every row."""
        m = getattr(self, "_pmask", None)
        d2, h2 = (delta ** 2).sum(-1), (h ** 2).sum(-1)
        if h.shape[1] > 1 and m is not None and tuple(m.shape) == tuple(h.shape[:2]):
            mf = m.float()
            d2, h2, n = (d2 * mf).sum(), (h2 * mf).sum(), mf.sum()
        else:
            d2, h2, n = d2.sum(), h2.sum(), d2.numel()
        rec = meter.setdefault(f"{'prefill' if h.shape[1] > 1 else 'decode'}:{key}", [0.0, 0.0, 0.0])
        rec[0] += float(d2); rec[1] += float(h2); rec[2] += float(n)

    def _install(self, direction, mode, layers, alpha, subspace=False, dose=None, meter=None):
        """Intervention hooks on decoder-layer outputs (and embedding output if layers=='all' and mode=='ablate').
        direction (RAW residual space): [D] one direction for all layers; [L,D] per-layer direction; [k,D] k-dim
        subspace for all layers (2-D arrays are read as [k,D] if subspace=True or shape[0]!=L); [L,k,D] per-layer
        subspace. Ablate: h <- h - (h @ Q) Q^T with Q = orthonormal basis (QR) of the layer's vectors.
        Steer: h <- h + alpha * u, u = first vector of the layer, unit-normalised (1-D only).
        dose (ablate only): None = full removal; scalar or per-layer [L] list c in [0,1]: h <- h - c (h @ Q) Q^T (the
        embedding hook uses dose[0]). meter: dict filled by _meter with the achieved perturbation per location."""
        import torch
        d = torch.tensor(direction, dtype=torch.float32, device="cuda")
        if d.ndim == 1:
            d = d[None, None]                      # [1,1,D]
        elif d.ndim == 2:
            d = d[None] if (subspace or d.shape[0] != self.L) else d[:, None]  # [1,k,D] or [L,1,D]
        assert d.ndim == 3 and d.shape[-1] == self.D and d.shape[0] in (1, self.L), tuple(d.shape)
        Qs = [torch.linalg.qr(v.T)[0] for v in d]  # each [D,k], orthonormal columns
        us = [v[0] / v[0].norm() for v in d]
        emb = layers in (None, "all") or (not isinstance(layers, str) and "emb" in list(layers))
        lidx = list(range(self.L)) if layers in (None, "all") else [int(x) for x in layers if x != "emb"]
        pick = (lambda l: 0) if d.shape[0] == 1 else (lambda l: l)
        dz = [1.0] * self.L if dose is None else ([float(dose)] * self.L if isinstance(dose, (int, float)) else [float(x) for x in dose])
        assert len(dz) == self.L and all(0.0 <= x <= 1.0 for x in dz), "dose must be in [0,1] per layer"

        def fn(h, l, key=None):
            hf = h.float()
            if mode == "ablate":
                Q = Qs[pick(l)]
                delta = dz[l] * ((hf @ Q) @ Q.T)
                if meter is not None:
                    self._meter(meter, l if key is None else key, hf, delta)
                hf = hf - delta
            elif mode == "steer":
                hf = hf + alpha * us[pick(l)]
            return hf.to(h.dtype)

        def hook(m, i, o, l):
            if isinstance(o, tuple):
                return (fn(o[0], l),) + o[1:]
            return fn(o, l)
        hs = [self.layers[l].register_forward_hook(lambda m, i, o, l=l: hook(m, i, o, l)) for l in lidx]
        if mode == "ablate" and emb:  # embedding OUTPUT hooked with the LAYER-0 direction (fitted on layer-0 output)
            hs.append(self.model.model.embed_tokens.register_forward_hook(lambda m, i, o: fn(o, 0, "emb")))
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
        self.last_arc_items = [dict(i=i, pred=p, gold=g, output=o) for i, (p, g, o) in enumerate(zip(pred, golds, outs))]
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

    @modal.method()
    def intervene_many(self, configs, ids, conditions=("C1", "C6"), do_arc=True, group="loso", messages_path=""):
        """Post-hoc LOSO robustness check (notes/loso_robustness_plan.md). Evaluates several ablation configs on the SAME
        episode subset in the same order (=> identical batching across configs). configs: list of dict(tag, direction)
        with direction None (matched baseline) or a RAW-space array as in _install; ablation at all layers + embedding.
        Behaviour only (no activations). Each result also goes to /vol/<group>/<tag>/metrics.json.
        messages_path: frozen messages jsonl on the volume (default: the original 7B set /vol/data/messages.jsonl)."""
        return self._intervene_many(configs, ids, conditions, do_arc, group, messages_path)

    def _intervene_many(self, configs, ids, conditions=("C1", "C6"), do_arc=True, group="loso", messages_path=""):
        import numpy as np
        import env
        mp = messages_path or f"/vol{_vp('')}/data/messages.jsonl"
        allrows = {r["id"]: r for r in (json.loads(l) for l in open(mp))}
        missing = [i for i in ids if i not in allrows]
        assert not missing, f"episodes missing from frozen messages: {missing}"
        rows = [allrows[i] for i in ids]
        msgs = {r["id"]: r for r in rows}
        out = []
        for cfg in configs:
            d = cfg["direction"]
            lay = cfg.get("layers", "all")  # "all" = all decoder layers + embedding; list = those decoder layers only
            meter = {} if cfg.get("meter") else None
            hs = [] if d is None else self._install(np.asarray(d, np.float32), "ablate", lay, 0.0, dose=cfg.get("dose"),
                                                     meter=meter)
            t0 = time.time()
            res = dict(tag=cfg["tag"], mode="none" if d is None else "ablate", layers=lay, n=len(rows), ids=list(ids),
                       model=self.model_id, conditions=list(conditions), messages_path=mp, dose=cfg.get("dose"),
                       meta=cfg.get("meta"))
            try:
                for c in conditions:
                    rr, _ = self._receivers(rows, msgs, c)
                    if meter is not None:
                        res.setdefault("meter", {})[c] = dict(meter)
                        meter.clear()
                    res[c] = env.decode_metrics(rr)
                    res[c]["decoded_dist"] = {str(k): sum(x["decoded"] == k for x in rr) for k in env.STATES + [None]}
                    res[c + "_rows"] = rr
                res["recall"], res["recall_acc"] = self._recall(rows, msgs)
                if do_arc:
                    res["arc_acc"] = self._arc(100)
                    res["arc_items"] = self.last_arc_items
            finally:
                for h in hs:
                    h.remove()
            # guard: no intervention hook may leak into the next config
            assert not any(lay._forward_hooks for lay in self.layers), "leaked layer hook"
            assert not self.model.model.embed_tokens._forward_hooks, "leaked embedding hook"
            res["seconds"] = time.time() - t0
            os.makedirs(f"/vol/{group}/{cfg['tag']}", exist_ok=True)
            json.dump(res, open(f"/vol/{group}/{cfg['tag']}/metrics.json", "w"))
            out.append(res)
        vol.commit()
        return out


    @modal.method()
    def fresh_set(self, gen_seed, configs, do_arc=True):
        """Fresh-message robustness (notes/followups_plan.md A): new COLLUDING sender messages for the same 306 episode
        contexts (env.episodes(range(17))), same prompts/sampling, sampling seed = gen_seed (original set: 0). IDs are
        m<gen_seed>_<orig_id>. Writes /vol/fresh/m<gen_seed>/messages.jsonl, then evaluates configs on all 306."""
        import env
        eps = env.episodes(range(17))
        texts = [self._tmpl(env.sender_msgs(e, True)) for e in eps]
        outs = self._generate(texts, GEN["sender_max_new"], temperature=GEN["sender_temperature"],
                              top_p=GEN["sender_top_p"], seed=gen_seed)
        rows = []
        for e, o in zip(eps, outs):
            r = dict(e, orig_id=e["id"], id=f"m{gen_seed}_{e['id']}", gen_seed=gen_seed, collude_raw=o,
                     collude_msg=env.parse_talk(o) or "Good luck everyone.", collude_parse_ok=bool(env.parse_talk(o)))
            r["collude_check"] = env.sender_check(r, r["collude_msg"])
            r["collude_clean"] = r["collude_check"]["clean_target"]
            rows.append(r)
        d = f"/vol/fresh/m{gen_seed}"
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/messages.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        vol.commit()
        res = self._intervene_many(configs, [r["id"] for r in rows], ("C1", "C6"), do_arc, f"fresh/m{gen_seed}",
                                   f"{d}/messages.jsonl")
        return dict(messages=rows, results=res)

    @modal.method()
    def baseline_and_extract(self, conditions=("C1", "C2", "C4", "C6"), extract_conds=("C1", "C4"),
                             messages_path="", out_dir="/vol/qwen32b/acts", group="qwen32b/baseline"):
        """32B step C1 (notes/followups_plan.md): matched no-intervention baseline on all frozen messages, then
        activation extraction (mean/last/full_last) for extract_conds only, using the baseline receiver outputs."""
        mp = messages_path or f"/vol{_vp('')}/data/messages.jsonl"
        rows = [json.loads(l) for l in open(mp)]
        base = self._intervene_many([dict(tag="baseline", direction=None)], [r["id"] for r in rows], conditions,
                                    True, group, mp)[0]
        outputs = {c: {x["id"]: x["output"] for x in base[c + "_rows"]} for c in extract_conds}
        t0 = time.time()
        meta = self._extract(rows, outputs, tuple(extract_conds), 0, out_dir)
        return dict(baseline=base, extract_seconds=time.time() - t0, meta_keys=list(meta))

    def _capture_last(self, rows, cond="C1"):
        """Prompt-only forward pass; returns fp16 [N, L, D] decision-token (`last`) activations at decoder-layer outputs.
        Any installed intervention hooks act during this pass (capture hooks are registered after them)."""
        import env
        colluding_rx, mkey = COND[cond]
        texts, spans, plens = [], [], []
        for r in rows:
            t = self._tmpl(env.receiver_msgs(_ep(r, cond), r[mkey], colluding_rx))
            sp = self._span(t, f"\n{r['sender']}: ", r[mkey])
            texts.append(t), spans.append(sp), plens.append(sp[2])
        return self._acts(texts, spans, plens)[1]

    @modal.method()
    def redund_fold(self, cfgs, ids, captures, cap_ids, group, messages_path=""):
        """Behavioural stage of notes/redundancy_plan.md §5 for one held-out fold: behavioural configs on the fold's
        held-out episodes (C1, C6, recall, ARC), then activation captures (dict(tag, direction|None, layers)) on
        cap_ids (all 306), saved as fp16 `last` arrays to /vol/<group>/cap_<tag>.npy."""
        import numpy as np
        res = self._intervene_many(cfgs, ids, ("C1", "C6"), True, group, messages_path)
        mp = messages_path or f"/vol{_vp('')}/data/messages.jsonl"
        allrows = {r["id"]: r for r in (json.loads(l) for l in open(mp))}
        rows = [allrows[i] for i in cap_ids]
        caps = {}
        for cap in captures:
            d = cap["direction"]
            hs = [] if d is None else self._install(np.asarray(d, np.float32), "ablate", cap["layers"], 0.0)
            t0 = time.time()
            try:
                last = self._capture_last(rows)
            finally:
                for h in hs:
                    h.remove()
            assert not any(lay._forward_hooks for lay in self.layers), "leaked layer hook"
            assert not self.model.model.embed_tokens._forward_hooks, "leaked embedding hook"
            os.makedirs(f"/vol/{group}", exist_ok=True)
            np.save(f"/vol/{group}/cap_{cap['tag']}.npy", last)
            caps[cap["tag"]] = dict(path=f"/{group}/cap_{cap['tag']}.npy", shape=list(last.shape),
                                    layers=cap["layers"], seconds=time.time() - t0)
        vol.commit()
        return dict(results=res, captures=caps)

    @modal.method()
    def validate_hooks(self, direction, configs, ids, messages_path="", pipeline_rows=()):
        """Hook-location validation (notes/unbundle_plan.md). For each config (dict(tag, layers|None)), run C1 generation on
        `ids` with PROBE hooks registered AFTER the intervention hooks on the embedding output and every decoder-layer
        output. Each probe records, per forward call, the phase (prefill: T>1, decode: T==1), the projection ratio
        ||hQ||/||h|| onto that location's span (embedding uses the layer-0 span), and float64 sum/sum-of-squares
        fingerprints, to compare against the matched unablated run. Also re-extracts `pipeline_rows` episodes' decision-
        token activations via the prompt-only and full-transcript pipelines, to explain capture-vs-saved differences."""
        import numpy as np
        import torch
        import env
        mp = messages_path or f"/vol{_vp('')}/data/messages.jsonl"
        allrows = {r["id"]: r for r in (json.loads(l) for l in open(mp))}
        rows = [allrows[i] for i in ids]
        d = torch.tensor(np.asarray(direction, np.float32), device="cuda")
        Qs = [torch.linalg.qr(v.T)[0] for v in d]                       # [L] of [D,k]
        out = {}
        for cfg in configs:
            rec = []

            def probe(name, l):
                def f(m, i, o):
                    h = (o[0] if isinstance(o, tuple) else o).float()
                    Q = Qs[l]
                    ratio = ((h @ Q).norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-9)).mean().item()
                    rec.append(dict(loc=name, phase="prefill" if h.shape[1] > 1 else "decode", ratio=ratio,
                                    s1=h.double().sum().item(), s2=(h.double() ** 2).sum().item()))
                return f
            hs = [] if cfg["layers"] is None else self._install(np.asarray(direction, np.float32), "ablate", cfg["layers"], 0.0,
                                                                 dose=cfg.get("dose"))
            ps = [self.model.model.embed_tokens.register_forward_hook(probe("emb", 0))]
            ps += [lay.register_forward_hook(probe(f"L{l}", l)) for l, lay in enumerate(self.layers)]
            try:
                texts = [self._tmpl(env.receiver_msgs(r, r["collude_msg"], True)) for r in rows]
                gen = self._generate(texts, GEN["decode_max_new"])
            finally:
                for h in hs + ps:
                    h.remove()
            assert not any(lay._forward_hooks for lay in self.layers) and not self.model.model.embed_tokens._forward_hooks
            out[cfg["tag"]] = dict(records=rec, outputs=gen)
        pipe = {}
        for name, batch in (pipeline_rows or {}).items():  # {name: [dict(id, output)]} = one original extraction batch
            prow = [allrows[x["id"]] for x in batch]
            texts, fulls, spans, plens = [], [], [], []
            for r, x in zip(prow, batch):
                t = self._tmpl(env.receiver_msgs(r, r["collude_msg"], True))
                sp = self._span(t, f"\n{r['sender']}: ", r["collude_msg"])
                texts.append(t), fulls.append(t + x["output"] + self.eot), spans.append(sp), plens.append(sp[2])
            prompt_only = self._acts(texts, spans, plens)[1]          # capture pipeline (prompt only)
            full = self._acts(fulls, spans, plens)[1]                 # original extraction pipeline (full transcript)
            pipe[name] = dict(ids=[x["id"] for x in batch], prompt_only=prompt_only.tolist(), full_transcript=full.tolist(),
                              full_len=[len(self.tok(f, add_special_tokens=False)["input_ids"]) for f in fulls],
                              prompt_len=plens)
        return dict(configs=out, pipeline=pipe)

    # ------------------------------------------------------------------ verification extension (notes/verify_plan.md)
    @modal.method()
    def run_jobs(self, jobs):
        """Several _intervene_many jobs sequentially in ONE container (one model load). job = dict(cfgs, ids, conditions,
        do_arc, group, messages_path)."""
        return [self._intervene_many(j["cfgs"], j["ids"], tuple(j["conditions"]), j["do_arc"], j["group"],
                                     j["messages_path"]) for j in jobs]

    @modal.method()
    def dose_fit(self, subspaces, layers, ids, messages_path=""):
        """Dose-fit pass (UNABLATED, prompt-only, right padding). For the C1 receiver prompt of each episode and each
        subspace {name: [L,k,D] or [L,D]} at each layer in `layers`: sum over non-pad prompt tokens of ||hQ||^2
        (E_all) and the same excluding position 0 and the decision token (E_mid); sum ||h||^2; token count."""
        import numpy as np
        import torch
        import env
        mp = messages_path or f"/vol{_vp('')}/data/messages.jsonl"
        allrows = {r["id"]: r for r in (json.loads(l) for l in open(mp))}
        rows = [allrows[i] for i in ids]
        texts = [self._tmpl(env.receiver_msgs(r, r["collude_msg"], True)) for r in rows]
        Q = {n: [torch.linalg.qr(torch.tensor(np.asarray(v[l], np.float32), device="cuda").reshape(-1, self.D).T)[0]
                 for l in layers] for n, v in subspaces.items()}
        N, nl = len(rows), len(layers)
        E = {n: np.zeros((N, nl, 2)) for n in Q}
        H, cnt = np.zeros((N, nl)), np.zeros(N)
        cap = {}
        hs = [self.layers[l].register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(l, o[0] if isinstance(o, tuple) else o))
              for l in layers]
        self.tok.padding_side = "right"
        try:
            for b0 in range(0, N, self.acts_bs):
                enc = self.tok(texts[b0:b0 + self.acts_bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    self.model.model(**enc)
                    m = enc["attention_mask"].float()
                    lens = m.sum(1).long()
                    mx = m.clone()
                    mx[:, 0] = 0
                    mx[torch.arange(len(lens)), lens - 1] = 0
                    b1 = b0 + len(lens)
                    for j, l in enumerate(layers):
                        h = cap[l].float()
                        H[b0:b1, j] = ((h ** 2).sum(-1) * m).sum(1).cpu().numpy()
                        for n in Q:
                            pr = ((h @ Q[n][j]) ** 2).sum(-1)
                            E[n][b0:b1, j, 0] = (pr * m).sum(1).cpu().numpy()
                            E[n][b0:b1, j, 1] = (pr * mx).sum(1).cpu().numpy()
                    cnt[b0:b1] = lens.cpu().numpy()
                cap.clear()
        finally:
            for h in hs:
                h.remove()
        assert not any(lay._forward_hooks for lay in self.layers)
        return dict(ids=list(ids), layers=list(layers), E={n: v.tolist() for n, v in E.items()}, H=H.tolist(), n=cnt.tolist())

    @modal.method()
    def capture_set(self, ids, messages_path, out_path):
        """Unablated prompt-only decision-token (`last`) capture of the C1 receiver prompts -> fp16 npy on the volume."""
        import numpy as np
        allrows = {r["id"]: r for r in (json.loads(l) for l in open(messages_path))}
        last = self._capture_last([allrows[i] for i in ids])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, last)
        vol.commit()
        return dict(path=out_path[len("/vol"):], shape=list(last.shape))

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
              arc: bool = True, acts: bool = True, subspace: bool = False, pull: bool = True, out_tag: str = "",
              gpu: str = "L40S"):
    """direction_file: local .npy [D] or [L,D]; 'random' -> random unit vector (seed 0); '' with mode=none."""
    import numpy as np
    if direction_file == "random":
        d = "random"
    elif direction_file:
        d = np.load(direction_file).astype(np.float32)
    else:
        d = None
    lay = "all" if layers == "all" else [int(x) for x in layers.split(",")]
    res = _runner(model, gpu).intervene.remote(d, mode, lay, alpha, tuple(conditions.split(",")), n or None, tag,
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


LOSO_GLOBAL = dict(dm="detect_C1vC4_mean_dm", pc1d="pc_ctrl_1d", state="state_C1_last_2d", pc2d="pc_ctrl_2d",
                   lr="detect_C1vC4_mean_lr")


def _runner(model, gpu):
    return (Runner if gpu == "L40S" else Runner.with_options(gpu=gpu))(model_id=model)


@app.local_entrypoint()
def loso(folds: str = "0,1,2,3,4,5", configs: str = "", smoke: int = 0, arc: bool = True, out_tag: str = "",
         model: str = DEFAULT_MODEL, gpu: str = "L40S", messages: str = ""):
    """Post-hoc LOSO robustness check (notes/loso_robustness_plan.md). One container per held-out scheme; every config
    (none, <dir>_global, <dir>_fold) is evaluated on that scheme's held-out episodes. Directions: data/directions/
    (global, fitted on all schemes) and data/loso/fold<k>/ (fitted without scheme k; exp/loso_fit.py).
    Outputs: data/loso/runs/<tag>/metrics.json. --smoke N evaluates only the first N held-out episodes (tag prefix smoke_).
    Other models (notes/followups_plan.md): --out-tag qwen14b|qwen32b --model ... --gpu ... [--messages /vol/...]:
    splits/global/fold directions from data_models/<tag>/loso/{splits.json,global,fold<k>}, outputs to
    data_models/<tag>/loso/runs. --messages defaults to /vol/<tag>/data/messages.jsonl (14B: its own frozen set)."""
    import numpy as np
    base = ROOT / (f"data_models/{out_tag}/loso" if out_tag else "data/loso")
    splits = json.load(open(base / "splits.json"))
    names = configs.split(",") if configs else ["none"] + [f"{b}_{w}" for b in LOSO_GLOBAL for w in ("global", "fold")]
    mp = messages or (f"/vol/{out_tag}/data/messages.jsonl" if out_tag else "")
    group = f"{out_tag}/loso" if out_tag else "loso"
    jobs = []
    for k in folds.split(","):
        ids = splits[k]["eval_ids"][: smoke or None]
        cfgs = []
        for nm in names:
            if nm == "none":
                d = None
            else:
                b, w = nm.rsplit("_", 1)
                if out_tag:
                    f = base / (f"global/{b}.npy" if w == "global" else f"fold{k}/{b}.npy")
                else:
                    f = ROOT / (f"data/directions/{LOSO_GLOBAL[b]}.npy" if w == "global" else f"data/loso/fold{k}/{b}.npy")
                d = np.load(f).astype(np.float32)
            cfgs.append(dict(tag=f"{'smoke' if smoke else 'loso'}_f{k}_{nm}", direction=d))
        jobs.append((cfgs, ids, ("C1", "C6"), arc, group, mp))
    t0 = time.time()
    for res_list in _runner(model, gpu).intervene_many.starmap(jobs):
        for res in res_list:
            od = base / "runs" / res["tag"]
            od.mkdir(parents=True, exist_ok=True)
            json.dump(res, open(od / "metrics.json", "w"), indent=1)
            print(res["tag"], "C1", round(res["C1"]["decode_acc"], 3), "C6 bet", round(res["C6"]["bet_exact_acc"], 3),
                  "n", res["n"], f"{res['seconds']:.0f}s", flush=True)
    print(f"wall {time.time() - t0:.0f}s")


@app.local_entrypoint()
def fresh(sets: str = "101,202,303,404,505", arc: bool = True):
    """Fresh-message robustness on Qwen2.5-7B (notes/followups_plan.md A). One container per sender seed set; FROZEN
    original directions (data/directions) for configs none, dm, pc1d, state, pc2d, lr; all 306 episodes per set.
    Outputs: data/fresh/m<seed>/messages.jsonl and data/fresh/m<seed>/runs/<tag>/metrics.json."""
    import numpy as np
    cfgs = [dict(tag="none", direction=None)] + [
        dict(tag=b, direction=np.load(ROOT / f"data/directions/{f}.npy").astype(np.float32)) for b, f in LOSO_GLOBAL.items()]
    seeds = [int(x) for x in sets.split(",")]
    t0 = time.time()
    for seed, out in zip(seeds, Runner().fresh_set.starmap([(s, cfgs, arc) for s in seeds])):
        od = ROOT / f"data/fresh/m{seed}"
        _write_jsonl(od / "messages.jsonl", out["messages"])
        for res in out["results"]:
            rd = od / "runs" / res["tag"]
            rd.mkdir(parents=True, exist_ok=True)
            json.dump(res, open(rd / "metrics.json", "w"), indent=1)
            print(f"m{seed}", res["tag"], "C1", round(res["C1"]["decode_acc"], 3), "C6 bet",
                  round(res["C6"]["bet_exact_acc"], 3), f"{res['seconds']:.0f}s", flush=True)
    print(f"wall {time.time() - t0:.0f}s")


@app.local_entrypoint()
def stage32(model: str = "Qwen/Qwen2.5-32B-Instruct", gpu: str = "H200", out_tag: str = "qwen32b"):
    """32B step C1 (notes/followups_plan.md C): baseline on the ORIGINAL 7B frozen messages (same episode ids/prompts/
    scoring) for C1, C2, C4, C6 + recall + ARC; then extract C1/C4 activations to /vol/<tag>/acts and pull locally."""
    import subprocess
    t0 = time.time()
    out = _runner(model, gpu).baseline_and_extract.remote(messages_path="/vol/data/messages.jsonl",
                                                          out_dir=f"/vol/{out_tag}/acts", group=f"{out_tag}/baseline")
    od = _local(out_tag)
    (od / "acts").mkdir(parents=True, exist_ok=True)
    (od / "baseline").mkdir(parents=True, exist_ok=True)
    json.dump(out["baseline"], open(od / "baseline/metrics.json", "w"), indent=1)
    b = out["baseline"]
    print({c: (round(b[c]["decode_acc"], 3), round(b[c]["bet_exact_acc"], 3)) for c in ("C1", "C2", "C4", "C6")},
          "recall", b["recall_acc"], "arc", b.get("arc_acc"), "extract s", round(out["extract_seconds"]), flush=True)
    for f in ("C1.npz", "C4.npz", "meta.json"):
        subprocess.run([str(ROOT / ".venv/bin/modal"), "volume", "get", "--force", "cutwire-vol",
                        f"/{out_tag}/acts/{f}", str(od / "acts")], check=True)
    print(f"wall {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------- follow-ups (notes/followups_plan.md)
@app.function(volumes={"/vol": vol}, timeout=3600, cpu=4, memory=8192)
def prefetch_weights(model_id: str):
    """CPU-only download of model weights into the volume HF cache (no GPU billed while downloading)."""
    from huggingface_hub import snapshot_download
    t0 = time.time()
    p = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
    vol.commit()
    size = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
    return dict(path=p, gb=size / 1e9, seconds=time.time() - t0)


@app.local_entrypoint()
def prefetch(model: str = "Qwen/Qwen2.5-32B-Instruct"):
    print(prefetch_weights.remote(model))



@app.local_entrypoint()
def redund(which: str = "7b", folds: str = "0,1,2,3,4,5", smoke: int = 0):
    """Core behavioural matrix of notes/redundancy_plan.md §5 (none, M2_all, M2_late, PC2_all, PC2_late) with
    activation captures (M2_all, M2_late on all 306 episodes per fold; unablated capture in fold 0 as a pipeline check).
    Frozen: fold directions (state.npy = M2, pc2d.npy = PC2), 7B message set, placement P_late = l >= ceil(2(L-1)/3).
    Outputs: data/redund/<which>/runs/<tag>/metrics.json and data/redund/<which>/caps/*.npy."""
    import math
    import subprocess
    import numpy as np
    spec = {"7b": ("Qwen/Qwen2.5-7B-Instruct", "L40S", ROOT / "data/loso", ""),
            "32b": ("Qwen/Qwen2.5-32B-Instruct", "H200", ROOT / "data_models/qwen32b/loso", "/vol/data/messages.jsonl")}
    model, gpu, base, mp = spec[which]
    splits = json.load(open(base / "splits.json"))
    all_ids = [json.loads(l)["id"] for l in open(ROOT / "data/messages.jsonl")]
    jobs = []
    for k in folds.split(","):
        M2 = np.load(base / f"fold{k}/state.npy").astype(np.float32)
        PC2 = np.load(base / f"fold{k}/pc2d.npy").astype(np.float32)
        L = M2.shape[0]
        late = list(range(math.ceil(2 * (L - 1) / 3), L))
        pre = "smoke" if smoke else "redund"
        cfgs = [dict(tag=f"{pre}_f{k}_none", direction=None),
                dict(tag=f"{pre}_f{k}_M2_all", direction=M2, layers="all"),
                dict(tag=f"{pre}_f{k}_M2_late", direction=M2, layers=late),
                dict(tag=f"{pre}_f{k}_PC2_all", direction=PC2, layers="all"),
                dict(tag=f"{pre}_f{k}_PC2_late", direction=PC2, layers=late)]
        caps = [dict(tag=f"{pre}_f{k}_M2_all", direction=M2, layers="all"),
                dict(tag=f"{pre}_f{k}_M2_late", direction=M2, layers=late)]
        if k == "0":
            caps.append(dict(tag=f"{pre}_none", direction=None, layers="all"))
        ids = splits[k]["eval_ids"][: smoke or None]
        cap_ids = all_ids[: smoke or None]
        jobs.append((cfgs, ids, caps, cap_ids, f"redund/{which}", mp))
    od = ROOT / f"data/redund/{which}"
    t0 = time.time()
    for out in _runner(model, gpu).redund_fold.starmap(jobs):
        for res in out["results"]:
            rd = od / "runs" / res["tag"]
            rd.mkdir(parents=True, exist_ok=True)
            json.dump(res, open(rd / "metrics.json", "w"), indent=1)
            print(res["tag"], "C1", round(res["C1"]["decode_acc"], 3), "C6 bet", round(res["C6"]["bet_exact_acc"], 3),
                  "layers", "all" if res["layers"] == "all" else f"{res['layers'][0]}-{res['layers'][-1]}", flush=True)
        (od / "caps").mkdir(parents=True, exist_ok=True)
        for tag, c in out["captures"].items():
            subprocess.run([str(ROOT / ".venv/bin/modal"), "volume", "get", "--force", "cutwire-vol", c["path"],
                            str(od / "caps")], check=True, capture_output=True)
            print("capture", tag, c["shape"], f"{c['seconds']:.0f}s", flush=True)
        json.dump(out["captures"], open(od / f"captures_{list(out['captures'])[0]}.json", "w"), indent=1)
    print(f"wall {time.time() - t0:.0f}s")


def _unbundle_cfgs(k, pre):
    """7B unbundling cells (notes/unbundle_plan.md): fold-k M2 state span (data/loso/fold<k>/state.npy, unchanged)."""
    import numpy as np
    M2 = np.load(ROOT / f"data/loso/fold{k}/state.npy").astype(np.float32)
    late = list(range(18, 28))
    return M2, [dict(tag=f"{pre}_f{k}_none", direction=None),
                dict(tag=f"{pre}_f{k}_M2_late", direction=M2, layers=late),
                dict(tag=f"{pre}_f{k}_M2_all", direction=M2, layers="all"),
                dict(tag=f"{pre}_f{k}_EMB_LATE", direction=M2, layers=["emb"] + late),
                dict(tag=f"{pre}_f{k}_PRE_LATE", direction=M2, layers=list(range(28)))]


@app.local_entrypoint()
def unbundle(folds: str = "0,1,2,3,4,5", smoke: int = 0):
    """7B placement unbundling: none, M2_late, M2_all (reproductions of the core matrix), EMB_LATE (embedding hook with
    the original layer-0 direction + layers 18-27), PRE_LATE (layers 0-27, no embedding hook). Held-out fold episodes,
    C1, C6, recall, ARC (per item). Outputs: data/unbundle/runs/<tag>/metrics.json."""
    splits = json.load(open(ROOT / "data/loso/splits.json"))
    pre = "smoke" if smoke else "unb"
    jobs = []
    for k in folds.split(","):
        _, cfgs = _unbundle_cfgs(k, pre)
        jobs.append((cfgs, splits[k]["eval_ids"][: smoke or None], ("C1", "C6"), True, "unbundle", ""))
    t0 = time.time()
    for res_list in Runner().intervene_many.starmap(jobs):
        for res in res_list:
            od = ROOT / "data/unbundle/runs" / res["tag"]
            od.mkdir(parents=True, exist_ok=True)
            json.dump(res, open(od / "metrics.json", "w"), indent=1)
            lay = res["layers"]
            print(res["tag"], "C1", round(res["C1"]["decode_acc"], 3), "C6", round(res["C6"]["bet_exact_acc"], 3),
                  "ARC", res.get("arc_acc"), "layers", lay if isinstance(lay, str) else f"{lay[0]}..{lay[-1]} ({len(lay)})",
                  flush=True)
    print(f"wall {time.time() - t0:.0f}s")


@app.local_entrypoint()
def unbundle_validate():
    """Hook-location validation for the unbundling cells (prefill + decode probes; fold-0 direction; 9 held-out
    fold-0 episodes) + capture-vs-saved pipeline test on extraction batches 4 (differing) and 9 (identical)."""
    splits = json.load(open(ROOT / "data/loso/splits.json"))
    M2, cfgs = _unbundle_cfgs("0", "val")
    vcfgs = [dict(tag=c["tag"].split("_f0_")[1], layers=c.get("layers")) for c in cfgs]
    ids = splits["0"]["eval_ids"][::6]
    meta = json.load(open(ROOT / "data/acts/meta.json"))
    outs = meta["conditions"]["C1"]["receiver_output"]
    batches = {f"batch{b}": [dict(id=meta["ids"][i], output=outs[i]) for i in range(16 * b, 16 * b + 16)] for b in (4, 9)}
    res = Runner().validate_hooks.remote(M2, vcfgs, ids, "", batches)
    od = ROOT / "data/unbundle/validation"
    od.mkdir(parents=True, exist_ok=True)
    json.dump(dict(ids=ids, configs=res["configs"]), open(od / "hooks.json", "w"))
    import numpy as np
    for name, p in res["pipeline"].items():
        np.savez(od / f"pipeline_{name}.npz", prompt_only=np.array(p["prompt_only"], np.float16),
                 full_transcript=np.array(p["full_transcript"], np.float16), full_len=p["full_len"], prompt_len=p["prompt_len"],
                 ids=np.array(p["ids"]))
    print("saved", od)


# ---------------------------------------------------------------------- verification extension (notes/verify_plan.md)
def _load_cfgs(cfgs, prefix=""):
    """Matrix cfg: dict(tag, dir=<repo-relative .npy>|None, layers='all'|list, dose=None|float|[L], meta=...)."""
    import numpy as np
    return [dict(tag=prefix + c["tag"], direction=None if c["dir"] is None else np.load(ROOT / c["dir"]).astype(np.float32),
                 layers=c.get("layers", "all"), dose=c.get("dose"), meter=c["dir"] is not None, meta=c.get("meta"))
            for c in cfgs]


@app.local_entrypoint()
def verify(matrix: str, smoke: int = 0):
    """Run a frozen matrix data/verify/matrices/<matrix>.json = dict(model, gpu, containers=[[job, ...], ...]); each
    container = one model load running its jobs sequentially. --smoke N: first job of the first container, first N ids,
    tags prefixed smoke_. Outputs: data/verify/runs/<matrix>/<tag>/metrics.json."""
    M = json.load(open(ROOT / f"data/verify/matrices/{matrix}.json"))
    pre = "smoke_" if smoke else ""
    conts = [[M["containers"][0][0]]] if smoke else M["containers"]
    jobs = [[dict(j, cfgs=_load_cfgs(j["cfgs"], pre), ids=j["ids"][: smoke or None]) for j in c] for c in conts]
    od = ROOT / f"data/verify/runs/{matrix}"
    t0 = time.time()
    for out in _runner(M["model"], M["gpu"]).run_jobs.map(jobs):
        for res_list in out:
            for res in res_list:
                rd = od / res["tag"]
                rd.mkdir(parents=True, exist_ok=True)
                json.dump(res, open(rd / "metrics.json", "w"), indent=1)
                lay = res["layers"]
                print(res["tag"], "C1", round(res["C1"]["decode_acc"], 3), "bet", round(res["C1"]["bet_exact_acc"], 3),
                      "C6", round(res["C6"]["bet_exact_acc"], 3), "ARC", res.get("arc_acc"),
                      "layers", lay if isinstance(lay, str) else f"{lay[0]}..{lay[-1]} ({len(lay)})", f"{res['seconds']:.0f}s",
                      flush=True)
    print(f"wall {time.time() - t0:.0f}s")


@app.local_entrypoint()
def vdosefit():
    """7B dose-fit pass: unablated C1 prompts of all 306 original episodes; fold M2 (state.npy) and PC2 (pc2d.npy)
    subspaces of all 6 folds at layers 16-27. Output data/verify/dose_fit.json (alpha is derived on CPU from TRAINING
    episodes only by exp/verify_matrix.py)."""
    import numpy as np
    ids = [json.loads(l)["id"] for l in open(ROOT / "data/messages.jsonl")]
    sub = {f"f{k}_{n}": np.load(ROOT / f"data/loso/fold{k}/{f}.npy").astype(np.float32)
           for k in range(6) for n, f in (("M2", "state"), ("PC2", "pc2d"))}
    t0 = time.time()
    res = Runner().dose_fit.remote(sub, list(range(16, 28)), ids, "")
    (ROOT / "data/verify").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(ROOT / "data/verify/dose_fit.json", "w"))
    print(f"dose fit done, wall {time.time() - t0:.0f}s")


@app.local_entrypoint()
def vvalidate(spec: str):
    """Hook validation for changed paths: data/verify/matrices/<spec>.json = dict(model, gpu, dir, ids, messages_path,
    configs=[dict(tag, layers|None, dose)]). Output data/verify/validation/<spec>.json."""
    import numpy as np
    S = json.load(open(ROOT / f"data/verify/matrices/{spec}.json"))
    res = _runner(S["model"], S["gpu"]).validate_hooks.remote(np.load(ROOT / S["dir"]).astype(np.float32), S["configs"],
                                                               S["ids"], S["messages_path"], {})
    od = ROOT / "data/verify/validation"
    od.mkdir(parents=True, exist_ok=True)
    json.dump(dict(spec=S, configs=res["configs"]), open(od / f"{spec}.json", "w"))
    print("saved", od / f"{spec}.json")
