"""CPU check (transformers 4.46.3, tiny random Qwen2 + Qwen2.5 Instruct generation_config): logits processors built for the S5
sampling kwargs. Run: /tmp/tfchk/bin/python -I exp/check_sampling_processors.py (venv with transformers==4.46.3, torch==2.5.1)."""
# Which logits processors does transformers 4.46.3 build inside generate() for our exact kwargs, given Qwen's default config?
import glob, torch
from transformers import AutoModelForCausalLM, Qwen2Config
from transformers.generation import GenerationConfig
snap = glob.glob("/Users/tanujd/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/")[0]
cfg = Qwen2Config(vocab_size=1000, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)
m = AutoModelForCausalLM.from_config(cfg)
m.generation_config = GenerationConfig.from_pretrained(snap)
m.generation_config.eos_token_id = [1]; m.generation_config.pad_token_id = 0
print("model defaults:", {k: getattr(m.generation_config, k) for k in ("do_sample","temperature","top_p","top_k","repetition_penalty")})
orig = m._get_logits_processor
seen = {}
def wrap(*a, **k):
    lp = orig(*a, **k); seen["lp"] = [(type(p).__name__, {kk: vv for kk, vv in vars(p).items() if kk in ("temperature","top_p","top_k","penalty")}) for p in lp]
    return lp
m._get_logits_processor = wrap
x = torch.tensor([[5, 6, 7, 8]])
for name, kw in (("sampled (S5)", dict(do_sample=True, temperature=0.7, top_p=1.0, top_k=None, repetition_penalty=1.0)),
                 ("greedy", dict(do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0))):
    m.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=3, pad_token_id=0, **kw)
    print(name, seen["lp"])
