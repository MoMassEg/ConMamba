# ==============================================================================
# EJUST Egyptian Arabic ASR — Q1 Missing Experiments & Fast Dual-T4 Runner
# - Skips already-completed conmamba_small (seed 7775: 13.85 / 14.65 WER)
# - Runs missing baselines (conformer_small, conmambamamba_small) + multi-seed
# - Evaluates 3 long-form decoding strategies (fixed, chunked, overlap_energy)
# - Optional fast Whisper baseline on test / test_long
# - Prints full Q1 tables, RTF, Multi-Seed Mean ± Std, and LaTeX (no figures)
# ==============================================================================

import os, sys, re, glob, json, time, shutil, platform, subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==============================================================================
# 0. EXPERIMENT CONFIGURATION (EDIT ONLY THIS BLOCK)
# ==============================================================================
EXPERIMENTS = [
    # (model_name, seed)
    ("conformer_small",     7775),
    ("conmambamamba_small", 7775),
    # Uncomment for full 3-seed Q1 statistical validation (runs 2 at a time on T4x2):
    # ("conformer_small",     42),
    # ("conmambamamba_small", 42),
    # ("conmamba_small",      42),
    # ("conformer_small",     1234),
    # ("conmambamamba_small", 1234),
    # ("conmamba_small",      1234),
]

EPOCHS = 110                    # Zevallos et al. (2025)
VALID_SEARCH_INTERVAL = 55      # Fast: beam-search valid only at ep 55 & 110
TIME_LIMIT_H = 11.2             # Safe stop before Kaggle's 12h hard limit
RUN_WHISPER_BASELINE = True     # Fast (~4 min) inference-only Whisper baseline
WHISPER_MODEL = "openai/whisper-small"  # Or "openai/whisper-large-v3"
LONG_STRATEGIES = ["fixed", "chunked", "overlap_energy"]  # Q1 Option 2 ablation
PREV_CONMAMBA_RESULT = {        # Already-completed run (included in final Q1 table)
    "model": "conmamba_small", "seed": 7775, "params_m": 12.1,
    "test_wer": 13.85, "test_cer": 4.44, "test_long_wer": 14.65, "test_long_cer": 4.57,
    "test_ctc_wer": 18.83, "test_long_ctc_wer": 72.94, "rtf_test": None, "rtf_long": None,
}
RESUME_INPUT_DIR = None         # Optional: "/kaggle/input/<prev-run-dataset>"

WORK = "/kaggle/working"
REPO = f"{WORK}/Mamba-ASR"
RESULTS_ROOT = f"{WORK}/q1_results"
os.makedirs(RESULTS_ROOT, exist_ok=True)

# ==============================================================================
# 1. ENVIRONMENT & DEPENDENCY SETUP
# ==============================================================================
import torch
NUM_GPUS = torch.cuda.device_count()
assert NUM_GPUS >= 1, "No GPU detected! Enable Accelerator -> GPU T4 x2."
print(f"Python {platform.python_version()} | PyTorch {torch.__version__} | GPUs: {NUM_GPUS}x {torch.cuda.get_device_name(0)}")

def pip_install(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)

pip_install(["speechbrain==1.0.3", "sentencepiece", "einops", "soundfile", "packaging", "ninja", "jiwer", "scipy"])
pip_install(["causal-conv1d", "mamba-ssm", "--no-build-isolation"])
if RUN_WHISPER_BASELINE:
    pip_install(["transformers", "accelerate"])

if not os.path.isdir(REPO):
    subprocess.check_call(["git", "clone", "-q", "https://github.com/xi-j/Mamba-ASR.git", REPO])

# ==============================================================================
# 2. WRITE PATCHED ejust_tools.py (FIXED YAML + SPEED + Q1 OVERRIDES)
# 4. LOCATE DATASET (ejust-prepared-v1)
# ==============================================================================
import pandas as pd
bpe_candidates = sorted(glob.glob("/kaggle/input/**/tokenizer/bpe500.model", recursive=True))
assert bpe_candidates, "ejust-prepared-v1 not found! Add Input -> Datasets -> ejust-prepared-v1."
BPE = bpe_candidates[0]
DATA = os.path.dirname(os.path.dirname(BPE))
print(f"Dataset root: {DATA}")

# ==============================================================================
# 5. PARALLEL DUAL-GPU TRAINING RUNNER (GPU 0 & GPU 1 SIMULTANEOUSLY)
# ==============================================================================
GLOBAL_T0 = time.time()
TIME_LIMIT_SEC = int(TIME_LIMIT_H * 3600)

def run_single_experiment(model_name, seed, gpu_id):
    run_id = f"{model_name}_s{seed}"
    out_dir = f"{RESULTS_ROOT}/{run_id}"
    done_marker = f"{out_dir}/wer_output/wer_test_long_{LONG_STRATEGIES[-1]}.txt"

    if RESUME_INPUT_DIR:
        prev = sorted(glob.glob(f"{RESUME_INPUT_DIR}/**/{run_id}", recursive=True))
        if prev and not os.path.isdir(out_dir):
            shutil.copytree(prev[0], out_dir)

    if os.path.exists(done_marker):
        print(f"[GPU {gpu_id}] SKIP {run_id} (already completed).")
        return run_id, 0

    hp_path = f"{WORK}/{run_id}.yaml"
    ejust_tools.make_hparams(
        f"{REPO}/hparams/S2S/{model_name}.yaml", hp_path,
        data_folder=DATA, csv_dir=DATA, bpe_model=BPE, output_folder=out_dir,
        epochs=EPOCHS, warmup=30000, valid_search_interval=VALID_SEARCH_INTERVAL,
        batch_size=32, grad_accumulation_factor=1, num_workers=2,
        long_strategies=",".join(LONG_STRATEGIES), skip_train=False,
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable, "train_egy_q1.py", hp_path,
        "--precision", "fp16", "--seed", str(seed), "--ckpt_interval_minutes", "30",
    ]
    print(f"[GPU {gpu_id}] START {run_id} -> {out_dir}")
    t0 = time.time()
    log_path = f"{WORK}/{run_id}_stdout.txt"
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            logf.write(line)
            if any(k in line for k in ("train_logger", "Epoch", "ERROR", "Traceback")):
                print(f"[{run_id}|GPU{gpu_id}] {line}", end="")
            if time.time() - GLOBAL_T0 > TIME_LIMIT_SEC:
                print(f"[GPU {gpu_id}] TIME LIMIT REACHED — terminating {run_id} safely (checkpoints saved).")
                proc.terminate()
                break
        proc.wait()
    print(f"[GPU {gpu_id}] DONE {run_id} | exit={proc.returncode} | {(time.time()-t0)/60:.1f} min")
    return run_id, proc.returncode

with ProcessPoolExecutor(max_workers=NUM_GPUS) as ex:
    futures = {}
    for idx, (m_name, s) in enumerate(EXPERIMENTS):
        gpu_id = idx % NUM_GPUS
        futures[ex.submit(run_single_experiment, m_name, s, gpu_id)] = (m_name, s)
    for fut in as_completed(futures):
        rid, rc = fut.result()

# ==============================================================================
# 6. OPTIONAL FAST WHISPER BASELINE (INFERENCE-ONLY ON test & test_long)
# ==============================================================================
whisper_row = None
if RUN_WHISPER_BASELINE and (time.time() - GLOBAL_T0 < TIME_LIMIT_SEC - 600):
    print(f"\n=== Running Inference-Only Baseline: {WHISPER_MODEL} ===")
    from jiwer import wer as jiwer_wer, cer as jiwer_cer
    from transformers import pipeline

    pipe = pipeline(
        "automatic-speech-recognition", model=WHISPER_MODEL,
        device=0, torch_dtype=torch.float16, chunk_length_s=30,
    )
    n_w_params = sum(p.numel() for p in pipe.model.parameters()) / 1e6
    w_stats = {"model": WHISPER_MODEL.split("/")[-1], "seed": 0, "params_m": round(n_w_params, 1)}

    for split in ["test", "test_long"]:
        df = pd.read_csv(f"{DATA}/{split}.csv", dtype=str, keep_default_na=False)
        wavs = [p.replace("$data_root", DATA) for p in df["wav"].tolist()]
        refs = df["wrd"].tolist()
        total_audio = df["duration"].astype(float).sum()
        t0 = time.time()
        outs = pipe(wavs, batch_size=16, generate_kwargs={"language": "arabic", "task": "transcribe"})
        elapsed = time.time() - t0
        hyps = [o["text"].strip() for o in outs]
        wer_key = f"{split}_wer"
        cer_key = f"{split}_cer"
        rtf_key = "rtf_test" if split == "test" else "rtf_long"
        w_stats[wer_key] = round(jiwer_wer(refs, hyps) * 100, 2)
        w_stats[cer_key] = round(jiwer_cer(refs, hyps) * 100, 2)
        w_stats[rtf_key] = round(elapsed / total_audio, 4)
        print(f"  {split:10s} | WER={w_stats[wer_key]:.2f}% | CER={w_stats[cer_key]:.2f}% | RTF={w_stats[rtf_key]:.4f}")
    whisper_row = w_stats
    del pipe; torch.cuda.empty_cache()

# ==============================================================================
# 7. Q1 STATISTICAL REPORTING (TABLES, MULTI-SEED MEAN±STD, LATEX — NO FIGURES)
# ==============================================================================
def parse_wer_file(path):
    if not os.path.exists(path):
        return None
    line = open(path, encoding="utf-8").readline()
    m = re.search(r"%WER\s+([\d\.]+)", line)
    return float(m.group(1)) if m else None

rows = [PREV_CONMAMBA_RESULT]
if whisper_row:
    rows.append(whisper_row)

long_ablation_rows = []
for m_name, seed in EXPERIMENTS:
    run_id = f"{m_name}_s{seed}"
    out_dir = f"{RESULTS_ROOT}/{run_id}"
    wdir = f"{out_dir}/wer_output"
    if not os.path.isdir(wdir):
        continue
    p_file = Path(out_dir, "params.json")
    params_m = round(json.loads(p_file.read_text())["params"] / 1e6, 1) if p_file.exists() else None
    eff_file = Path(out_dir, "efficiency.jsonl")
    rtf_map = {}
    if eff_file.exists():
        for line in eff_file.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("stage") == "TEST" and rec.get("test_name"):
                rtf_map[rec["test_name"]] = rec.get("rtf")

    best_long_strat = "overlap_energy" if "overlap_energy" in LONG_STRATEGIES else LONG_STRATEGIES[-1]
    rows.append({
        "model": m_name, "seed": seed, "params_m": params_m,
        "test_wer": parse_wer_file(f"{wdir}/wer_test.txt"),
        "test_cer": parse_wer_file(f"{wdir}/cer_test.txt"),
        "test_long_wer": parse_wer_file(f"{wdir}/wer_test_long_{best_long_strat}.txt"),
        "test_long_cer": parse_wer_file(f"{wdir}/cer_test_long_{best_long_strat}.txt"),
        "test_ctc_wer": parse_wer_file(f"{wdir}/wer_ctc_test.txt"),
        "test_long_ctc_wer": parse_wer_file(f"{wdir}/wer_ctc_test_long_{best_long_strat}.txt"),
        "rtf_test": rtf_map.get("test"),
        "rtf_long": rtf_map.get(f"test_long_{best_long_strat}"),
    })
    for strat in LONG_STRATEGIES:
        long_ablation_rows.append({
            "model": m_name, "seed": seed, "strategy": strat,
            "test_long_wer": parse_wer_file(f"{wdir}/wer_test_long_{strat}.txt"),
            "test_long_cer": parse_wer_file(f"{wdir}/cer_test_long_{strat}.txt"),
            "rtf_long": rtf_map.get(f"test_long_{strat}"),
        })

df_main = pd.DataFrame(rows)
print("\n" + "=" * 90)
print("TABLE 1: PER-RUN Q1 BENCHMARK RESULTS (EJUST Egyptian Arabic ASR)")
print("=" * 90)
print(df_main.to_string(index=False))

agg_rows = []
for m, grp in df_main.groupby("model", sort=False):
    def fmt(col):
        vals = grp[col].dropna()
        if len(vals) == 0:
            return "-"
        if len(vals) == 1:
            return f"{vals.iloc[0]:.2f}"
        return f"{vals.mean():.2f} ± {vals.std(ddof=1):.2f}"
    agg_rows.append({
        "Model": m,
        "Params (M)": grp["params_m"].dropna().iloc[0] if len(grp["params_m"].dropna()) else "-",
        "Seeds": len(grp),
        "test WER (%)": fmt("test_wer"),
        "test CER (%)": fmt("test_cer"),
        "test_long WER (%)": fmt("test_long_wer"),
        "test_long CER (%)": fmt("test_long_cer"),
        "RTF (long)": f"{grp['rtf_long'].dropna().mean():.4f}" if len(grp["rtf_long"].dropna()) else "-",
    })
df_agg = pd.DataFrame(agg_rows)
print("\n" + "=" * 90)
print("TABLE 2: MULTI-SEED AGGREGATED Q1 TABLE (Mean ± Std)")
print("=" * 90)
print(df_agg.to_string(index=False))

if long_ablation_rows:
    df_long = pd.DataFrame(long_ablation_rows)
    print("\n" + "=" * 90)
    print("TABLE 3: LONG-FORM DECODING STRATEGY ABLATION (Option 2 Contribution)")
    print("=" * 90)
    print(df_long.to_string(index=False))

print("\n" + "=" * 90)
print("LATEX BOOKTABS TABLE (COPY-PASTE INTO MANUSCRIPT)")
print("=" * 90)
print(r"\begin{table}[t]")
print(r"\centering")
print(r"\caption{Egyptian Arabic ASR benchmark on EJUST (\texttt{test} and \texttt{test\_long}).}")
print(r"\begin{tabular}{lcccccc}")
print(r"\toprule")
print(r"\textbf{Model} & \textbf{Params} & \textbf{test WER} & \textbf{test CER} & \textbf{long WER} & \textbf{long CER} & \textbf{RTF} \\")
print(r"\midrule")
for _, r in df_agg.iterrows():
    model_tex = str(r["Model"]).replace("_", "\\_")
    print(f"{model_tex} & {r['Params (M)']}M & {r['test WER (%)']} & {r['test CER (%)']} & {r['test_long WER (%)']} & {r['test_long CER (%)']} & {r['RTF (long)']} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")