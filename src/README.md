# ConMamba-ASR: Modular Python Source (`src/`)

This directory contains the production-grade, modular Python source code implementing dataset curation, acoustic modeling, training pipelines, and multi-GPU evaluation benchmarks for:

> **"ConMamba-ASR: Selective State Space Modeling and Acoustic Curation for Dialectal Egyptian Arabic Speech Recognition"**  
> *Authors: Rana Z. Dighedy, Mohamed S. Amer, Monay M. Gad, Manal M. Ghaith, and Ahmed B. Zaky (E-JUST)*  
> *Submitted to IEEE Transactions on Audio, Speech, and Language Processing (IEEE TASLP)*

For full benchmark tables, architectural details, and the research paper summary, see the [Master Repository README](../README.md).

---

## 📁 Source Module Manifest

| File | Type | Primary Role & Description |
| :--- | :--- | :--- |
| [`audit_dataset.py`](audit_dataset.py) | **Phase 1 Pipeline** | Comprehensive acoustic audit and resampling engine: 44.1 kHz float inspection, 16 kHz polyphase sinc conversion, episode-level headroom-gain attenuation ($g_e \le 0.99$, guaranteeing 0 clipped samples), deep ECAPA-TDNN speaker verification, Whisper alignment check, and continuous long-form audio synthesis ($<60\text{ s}$). |
| [`train_egy_q1.py`](train_egy_q1.py) | **Training Engine** | Core SpeechBrain sequence-to-sequence trainer supporting ConMamba, Conformer, and ConMambaMamba architectures. Implements joint CTC/Attention loss ($\alpha=0.3$), Noam learning rate annealing, dynamic item data pipelines, and real-time GPU efficiency tracking (`efficiency.jsonl`). |
| [`run_dual_t4_benchmark.py`](run_dual_t4_benchmark.py) | **Multi-GPU Orchestrator** | Non-colliding multi-GPU parallel scheduler (e.g., dual NVIDIA Tesla T4), long-form decoding strategy ablation (fixed vs chunked vs overlap-energy), zero-shot Whisper baselines, paired bootstrap statistical significance tests, and automated LaTeX table generation. |
| [`ejust_tools.py`](ejust_tools.py) | **Compatibility & Config** | C++/CUDA compatibility shims for `causal_conv1d` and `mamba_ssm`, strict HyperPyYAML configuration patching, and environment locks. |
| [`requirements.txt`](requirements.txt) | **Dependencies** | Pinned dependencies for PyTorch, SpeechBrain 1.0.3, Mamba-SSM, Transformers, and audio processing tools. |

---

## ⚡ Quickstart

### 1. Environment Installation

```bash
pip install -r requirements.txt
pip install causal-conv1d>=1.2.0 mamba-ssm>=1.2.0 --no-build-isolation
```

### 2. Run Dataset Curation (Phase 1)

```bash
export EJUST_INPUT="/path/to/raw/ejust/segments.csv"
export EJUST_OUT="/path/to/curated/ejust_audit_v1"
export EJUST_WORKERS=8

python audit_dataset.py
```

### 3. Model Training (Phase 2)

```bash
python train_egy_q1.py /path/to/conmamba_small.yaml \
    --data_folder=/path/to/curated/ejust_audit_v1 \
    --output_folder=./results/conmamba_small_s7775 \
    --precision=fp16 \
    --seed=7775
```

### 4. Full Benchmark Execution (Phase 3 on Dual GPUs)

```bash
python run_dual_t4_benchmark.py
```
