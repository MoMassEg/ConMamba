import json
from pathlib import Path

src_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\src")
src_dir.mkdir(parents=True, exist_ok=True)

# 1. Extract from notebookd5d790028a (1).ipynb
nb1_path = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\notebookd5d790028a (1).ipynb")
with open(nb1_path, "r", encoding="utf-8") as f:
    nb1 = json.load(f)

lines1 = nb1["cells"][0]["source"]

# Extract ejust_tools_code
ejust_tools_lines = []
in_tools = False
for line in lines1:
    if "ejust_tools_code = r'''" in line or 'ejust_tools_code = """' in line:
        in_tools = True
        continue
    if in_tools and (line.startswith("'''") or line.startswith('"""')):
        in_tools = False
        break
    if in_tools:
        ejust_tools_lines.append(line)

(src_dir / "ejust_tools.py").write_text("".join(ejust_tools_lines), encoding="utf-8")
print(f"Wrote ejust_tools.py ({len(ejust_tools_lines)} lines)")

# Extract train_script (train_egy_q1.py)
train_script_lines = []
in_train = False
for line in lines1:
    if "train_script = r'''" in line or 'train_script = """' in line or "train_script = '''" in line:
        in_train = True
        continue
    if in_train and (line.startswith("'''") or line.startswith('"""')):
        in_train = False
        break
    if in_train:
        train_script_lines.append(line)

(src_dir / "train_egy_q1.py").write_text("".join(train_script_lines), encoding="utf-8")
print(f"Wrote train_egy_q1.py ({len(train_script_lines)} lines)")

# Extract runner script
runner_lines = []
for i, line in enumerate(lines1):
    # omit the giant embedded strings to make runner clean
    if i < 68 or i > 542:
        runner_lines.append(line)

(src_dir / "run_dual_t4_benchmark.py").write_text("".join(runner_lines), encoding="utf-8")
print(f"Wrote run_dual_t4_benchmark.py ({len(runner_lines)} lines)")

# 2. Extract from phase1.ipynb (complete audit script)
nb_phase1_path = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\phase1.ipynb")
with open(nb_phase1_path, "r", encoding="utf-8") as f:
    nb_p1 = json.load(f)

p1_code_blocks = []
p1_code_blocks.append("#!/usr/bin/env python3\n")
p1_code_blocks.append('"""\nEJUST Egyptian Arabic Speech Dataset — Phase 1 Acoustic Audit & 16kHz Curation Pipeline\n"""\n')

for i, cell in enumerate(nb_p1["cells"]):
    if cell["cell_type"] == "code":
        code_str = "".join(cell["source"])
        # filter out kaggle ! commands
        clean_lines = []
        for l in code_str.splitlines():
            if l.strip().startswith("!") or l.strip().startswith("%"):
                clean_lines.append("# " + l)
            else:
                clean_lines.append(l)
        p1_code_blocks.append(f"\n# --- Stage {i} ---\n" + "\n".join(clean_lines) + "\n")

(src_dir / "audit_dataset.py").write_text("".join(p1_code_blocks), encoding="utf-8")
print(f"Wrote audit_dataset.py ({len(p1_code_blocks)} stages)")

# 3. Create requirements.txt
reqs = """# Core Deep Learning & Audio Frameworks
torch>=2.0.0
torchaudio>=2.0.0
speechbrain==1.0.3
soundfile>=0.12.0
sentencepiece>=0.2.0

# State Space Model (Mamba) Kernels
mamba-ssm>=1.2.0
causal-conv1d>=1.2.0
ninja
packaging

# Baseline & Foundation Models
transformers>=4.40.0
huggingface-hub

# Data Processing & Numerical
numpy>=1.23.0
pandas>=2.0.0
scipy>=1.10.0
"""
(src_dir / "requirements.txt").write_text(reqs, encoding="utf-8")
print("Wrote requirements.txt")

# 4. Create README.md for src
readme_src = """# ConMamba-ASR: Source Code Repository

This directory contains the complete, standalone Python source code implementing the dataset curation, acoustic modeling, training pipeline, and evaluation benchmarks described in the paper:

**"ConMamba-ASR: Selective State Space Acoustic Modeling and Corpus Curation for Low-Resource Dialectal Egyptian Arabic Speech Recognition"**
*Authors: Omar N. Mahmoud, Mohamed S. Amer, Ahmed B. Zaky (E-JUST)*

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `audit_dataset.py` | Complete Phase 1 dataset curation pipeline: 44.1 kHz float inspection, 16 kHz headroom-gain conversion, clipping elimination, ECAPA-TDNN speaker verification, Whisper alignment check, and synthetic long-form audio synthesis. |
| `train_egy_q1.py` | Core SpeechBrain training script: ConMamba / Conformer / ConMambaMamba Seq2Seq training, joint CTC/Attention objective, Noam learning rate annealing, and real-time efficiency logger (`efficiency.jsonl`). |
| `ejust_tools.py` | Hyperparameter generation, YAML configuration patching, and repository compatibility shims for Mamba-ASR. |
| `run_dual_t4_benchmark.py` | Multi-GPU parallel orchestrator: launches training jobs concurrently across GPUs (e.g., dual NVIDIA Tesla T4), evaluates long-form decoding strategies (chunked vs fixed vs overlap), runs zero-shot Whisper baselines, and generates statistical tables. |
| `requirements.txt` | Python dependencies. |

---

## Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
git clone https://github.com/xi-j/Mamba-ASR.git
```

### 2. Dataset Curation (Phase 1)
```bash
python audit_dataset.py
```

### 3. Model Training (Phase 2)
```bash
python train_egy_q1.py /path/to/conmamba_small.yaml --precision fp16 --seed 7775
```

### 4. Full Benchmark Execution (Dual-GPU)
```bash
python run_dual_t4_benchmark.py
```
"""
(src_dir / "README.md").write_text(readme_src, encoding="utf-8")
print("Wrote src/README.md")
