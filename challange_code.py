"""EP48 - Control an LLM with two levers: decoding settings and prompt templates.
    python ep48_generate.py "A pen costs 3 dollars. Ana buys 4 and pays 20. Change?"
    python ep48_generate.py --settings "Write a slogan for a tea shop"
pip install "transformers>=4.46" torch accelerate
"""
import re
import sys
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# ── 1. CONFIG ─────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"       # change if needed
PRESETS = {
    "greedy":   dict(do_sample=False),
    "balanced": dict(do_sample=True, temperature=0.7, top_p=0.9),
    "creative": dict(do_sample=True, temperature=1.1, top_p=0.95, top_k=100),
}

# ── 2. PROMPT TEMPLATES ───────────────────────────────────────────────────
SYSTEM = "You are a precise assistant. Be concise."
SHOTS = [("2 + 2 * 3", "8"), ("(7 - 3) * 5", "20")]
msg = lambda role, text: {"role": role, "content": text}

def zero_shot(q):
    return [msg("system", SYSTEM), msg("user", q)]

def few_shot(q):
    shots = [m for x, y in SHOTS for m in (msg("user", x), msg("assistant", y))]
    return [msg("system", SYSTEM), *shots, msg("user", q)]

def role_play(q):
    return [msg("system", "You are a patient maths teacher. " + SYSTEM), msg("user", q)]

def structured(q):
    fmt = 'Reply ONLY with JSON: {"answer": <number>, "reason": "<one sentence>"}'
    return [msg("system", SYSTEM), msg("user", f"{q}\n\n{fmt}")]

def chain_of_thought(q):
    ask = "Think step by step, then end with a line 'Answer: <value>'."
    return [msg("system", SYSTEM), msg("user", f"{q}\n\n{ask}")]

TEMPLATES = {f.__name__: f for f in (zero_shot, few_shot, role_play, structured, chain_of_thought)}

# ── 3. LOAD + GENERATE ────────────────────────────────────────────────────
def load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    return tok, AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")

def generate(tok, model, messages, preset="balanced", max_new_tokens=200, seed=0):
    set_seed(seed)
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens,
                         pad_token_id=tok.eos_token_id, **PRESETS[preset])
    return tok.decode(out[0, enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

# ── 4. SELF-CONSISTENCY ───────────────────────────────────────────────────
def final_answer(text):
    found = re.findall(r"Answer:\s*\$?(-?[\d.,]+)", text)
    return found[-1].rstrip(".,") if found else None

def self_consistency(tok, model, q, n=5):
    votes = [final_answer(generate(tok, model, chain_of_thought(q), "creative", 300, seed=s))
             for s in range(n)]
    return Counter(v for v in votes if v).most_common(1), votes

# ── 5. COMPARE ────────────────────────────────────────────────────────────
def compare_prompts(tok, model, question):
    for name, build in TEMPLATES.items():
        print(f"\n=== {name} ===\n{generate(tok, model, build(question), 'greedy')}")

def compare_settings(tok, model, prompt):
    for preset in PRESETS:
        print(f"\n=== {preset} ===\n{generate(tok, model, zero_shot(prompt), preset)}")

if __name__ == "__main__":
    args = sys.argv[1:]
    tok_, model_ = load()
    if args and args[0] == "--settings":
        compare_settings(tok_, model_, " ".join(args[1:]))
    else:
        question = " ".join(args) or "A pen costs 3 dollars. Ana buys 4 and pays 20. Change?"
        compare_prompts(tok_, model_, question)
        print("\nself-consistency:", self_consistency(tok_, model_, question))
