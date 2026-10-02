# EP48 — Text Generation & Prompt Engineering

**Same model. Completely different answers.**

Control any LLM with two levers:
1. Decoding settings (temperature, top-k, top-p, beam)
2. Prompt templates

98-line production-ready script with 3 presets + 5 battle-tested templates.

---

### Quick Start

```bash
pip install "transformers>=4.46" torch accelerate

# Compare the 3 decoding presets
python ep48_generate.py --settings "Write a slogan for a tea shop"

# Compare all 5 prompt templates
python ep48_generate.py "A pen costs 3 dollars. Ana buys 4 and pays 20. Change?"
