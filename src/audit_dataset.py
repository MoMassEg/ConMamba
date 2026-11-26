#!/usr/bin/env python3
"""
EJUST Egyptian Arabic Speech Dataset — Phase 1 Acoustic Audit & 16kHz Curation Pipeline
"""

# --- Stage 1 ---
# 1. Setup
import os, sys, re, io, json, glob, hashlib, random, platform, unicodedata, collections, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CFG = dict(
    version="ejust-audit-v1",
    seed=1234,
    out_dir=os.environ.get("EJUST_OUT", "/kaggle/working/ejust_audit_v1"),
    input_glob=os.environ.get("EJUST_INPUT", "/kaggle/input/**/segments.csv"),
    orig_sr=44100,
    target_sr=16000,
    n_workers=int(os.environ.get("EJUST_WORKERS", os.cpu_count() or 2)),
    frame_ms=25,             # frame size for silence / energy measures
    silence_dbfs=-50.0,      # frame is "silent" below this level
    edge_ms=100,             # head/tail window for join analysis
    clip_level=0.999,        # |x| >= this is a full-scale sample
    clip_min_run=3,          # clipping = a run of >= this many consecutive full-scale samples
                             # (files are peak-normalized to 1.0, so a single full-scale sample is normal)
    gain_headroom=0.99,      # per-episode gain so the *resampled* 16 kHz episode peak stays <= this
    min_dur_s=0.15,          # shorter clips are flagged
    cps_mad_k=4.0,           # speaking-rate band: median +- k * MAD on log(chars/sec)
    speaker_min_dur_s=1.0,   # embeddings of shorter clips are too noisy to judge
    speaker_mad_k=5.0,       # speaker outlier: distance > median + k * MAD
    run_speaker=os.environ.get("EJUST_RUN_SPEAKER", "1") == "1",
    run_whisper=os.environ.get("EJUST_RUN_WHISPER", "1") == "1",
    whisper_model="openai/whisper-large-v3",
    whisper_batch=16,
    cer_exclude=0.6,         # rule threshold for "suggest exclude" (only with another flag)
    join_review_n=20,        # joins written to joins_review.csv (half riskiest, half random)
)
random.seed(CFG["seed"]); np.random.seed(CFG["seed"])

OUT = Path(CFG["out_dir"]); OUT.mkdir(parents=True, exist_ok=True)
WAV16 = OUT / "wav16k"; WAV16.mkdir(exist_ok=True)
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)

seg_hits = glob.glob(CFG["input_glob"], recursive=True)
assert len(seg_hits) == 1, f"expected exactly one segments.csv, found {seg_hits}"
SEG_CSV = Path(seg_hits[0])
AUDIO_DIR = SEG_CSV.parent
print("segments.csv :", SEG_CSV)
print("audio folder :", AUDIO_DIR)

def lib_versions():
    v = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
         "soundfile": sf.__version__, "scipy": __import__("scipy").__version__}
    for m in ["torch", "transformers", "speechbrain"]:
        try:
            v[m] = str(__import__(m).__version__)      # torch's version is a str subclass yaml can't write
        except Exception:
            v[m] = None
    return v

import yaml, subprocess

def write_env_lock():
    """Full list of installed packages (pip freeze): the environment lock file."""
    lock = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (OUT / "environment_lock.txt").write_text(f"# python {platform.python_version()}\n" + lock, encoding="utf-8")

(OUT / "config.yaml").write_text(yaml.safe_dump({"cfg": CFG, "versions": lib_versions(), "segments_csv": str(SEG_CSV),
                                                 "started": time.strftime("%Y-%m-%d %H:%M:%S")},
                                                allow_unicode=True, sort_keys=False), encoding="utf-8")
write_env_lock()
with open(OUT / "config.json", "w", encoding="utf-8") as f:
    json.dump({"cfg": CFG, "versions": lib_versions(), "segments_csv": str(SEG_CSV),
               "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=2)
print(json.dumps(CFG, indent=1))

# --- Stage 2 ---
# 2. Load transcripts and match them to audio files
raw_bytes = SEG_CSV.read_bytes()
seg = pd.read_csv(io.StringIO(raw_bytes.decode("utf-8-sig")), header=None, names=["ID", "text"],
                  dtype=str, keep_default_na=False)
assert seg["ID"].is_unique, "duplicate IDs in segments.csv"

wav_files = sorted(AUDIO_DIR.glob("*.wav"))
def id_of(p):
    m = re.search(r"(\d+)\.wav$", p.name)
    return m.group(1) if m else None
wav_map = {id_of(p): p for p in wav_files}
assert None not in wav_map, "a wav file name has no numeric ID"

ids_text, ids_wav = set(seg["ID"]), set(wav_map)
print(f"transcripts: {len(seg)} | wav files: {len(wav_files)}")
print("text without audio:", sorted(ids_text - ids_wav)[:20], len(ids_text - ids_wav))
print("audio without text:", sorted(ids_wav - ids_text)[:20], len(ids_wav - ids_text))
nums = sorted(int(i) for i in ids_wav)
print("ID range", nums[0], "-", nums[-1], "| gaps:", len(set(range(nums[0], nums[-1] + 1)) - set(nums)))

seg = seg[seg["ID"].isin(ids_wav)].copy()
seg["order"] = seg["ID"].astype(int)
seg = seg.sort_values("order").reset_index(drop=True)
seg["wav_orig"] = seg["ID"].map(lambda i: str(wav_map[i]))

# --- Stage 3 ---
# 3. Audio pass A: measure every original file (read-only)
def frame_db(x, sr, frame_ms):
    n = max(1, int(sr * frame_ms / 1000))
    k = len(x) // n
    if k == 0:
        return np.array([20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-10)])
    fr = x[: k * n].reshape(k, n)
    return 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-10)

def to16k(x, sr):
    g = np.gcd(CFG["target_sr"], sr)
    return resample_poly(x.astype(np.float64), CFG["target_sr"] // g, sr // g)

def run_lengths(mask):
    """Lengths of the runs of True values in a boolean array."""
    if not mask.any():
        return np.zeros(0, dtype=int)
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)

def rms_db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-10)) if len(x) else -200.0

def measure(path):
    info = sf.info(path)
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    ch = x.shape[1]
    mono = x.mean(axis=1) if ch > 1 else x[:, 0]
    e = int(sr * CFG["edge_ms"] / 1000)
    fdb = frame_db(mono, sr, CFG["frame_ms"])
    active = fdb > CFG["silence_dbfs"]
    runs = run_lengths(np.abs(mono) >= CFG["clip_level"])
    return dict(
        sr=sr, channels=ch, subtype=info.subtype, format=info.format, frames=len(mono),
        duration=len(mono) / sr,
        peak=float(np.max(np.abs(mono))) if len(mono) else 0.0,
        peak16=float(np.max(np.abs(to16k(mono, sr)))) if len(mono) else 0.0,
        n_clip_runs=int(np.sum(runs >= CFG["clip_min_run"])),
        max_clip_run=int(runs.max()) if len(runs) else 0,
        n_nonfinite=int(np.sum(~np.isfinite(mono))),
        rms_db=rms_db(mono),
        active_db=float(np.mean(fdb[active])) if active.any() else -200.0,
        active_frac=float(active.mean()),
        noise_floor_db=float(np.percentile(fdb, 10)),     # level of the quietest 10% of frames
        head_db=rms_db(mono[:e]), tail_db=rms_db(mono[-e:]),
        dc=float(np.mean(mono)) if len(mono) else 0.0,
        sha256=hashlib.sha256(mono.astype("<f4").tobytes()).hexdigest(),
    )

def measure_safe(path):
    try:
        return dict(read_error="", **measure(path))
    except Exception as ex:
        return dict(read_error=repr(ex))

def pmap(fn, items):
    if CFG["n_workers"] <= 1:
        return list(map(fn, items))
    with ProcessPoolExecutor(CFG["n_workers"]) as pool:
        return list(pool.map(fn, items, chunksize=64))

t0 = time.time()
meas = pmap(measure_safe, seg["wav_orig"])
inv = pd.concat([seg, pd.DataFrame(meas)], axis=1)
print(f"pass A done in {time.time() - t0:.0f}s")
print("read errors:", (inv["read_error"] != "").sum())
print(inv[["sr", "channels", "subtype"]].value_counts())
print("total hours:", round(inv["duration"].sum() / 3600, 3))
print(inv["duration"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]))
print(f"clips with a clipping run (>= {CFG['clip_min_run']} samples at |x| >= {CFG['clip_level']}): "
      f"{int((inv['n_clip_runs'] > 0).sum())} | longest run: {int(inv['max_clip_run'].max())} samples")

# --- Stage 4 ---
# 4. Transcript audit (descriptive only; the official normalization is defined in Notebook 2)
ARABIC_DIAC = re.compile(r"[ً-ْٰ]")
INVISIBLE = re.compile(r"[﻿​-‏‪-‮⁦-⁩]")

def compare_norm(t):
    """Light normalization used ONLY to compare texts here (Whisper CER, duplicate detection)."""
    t = unicodedata.normalize("NFC", t)
    t = INVISIBLE.sub("", t).replace("ـ", "")
    t = ARABIC_DIAC.sub("", t)
    t = re.sub(r"[أإآ]", "ا", t).replace("ى", "ي")
    t = re.sub(r"[^ء-ي\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

txt = inv["text"]
inv["n_words"] = txt.map(lambda t: len(compare_norm(t).split()))
inv["n_chars"] = txt.map(lambda t: len(compare_norm(t).replace(" ", "")))
inv["cps"] = inv["n_chars"] / inv["duration"].clip(lower=1e-3)

chars = collections.Counter(ch for t in txt for ch in t)
char_tab = pd.DataFrame([(ch, f"U+{ord(ch):04X}", unicodedata.category(ch), unicodedata.name(ch, "?"), c)
                         for ch, c in chars.items()], columns=["char", "code", "category", "name", "count"]
                        ).sort_values("count", ascending=False)
char_tab.to_csv(OUT / "char_inventory.csv", index=False, encoding="utf-8-sig")

noise = {
    "empty_text": int((txt.str.strip() == "").sum()),
    "leading_trailing_space": int((txt != txt.str.strip()).sum()),
    "double_space": int(txt.str.contains("  ", regex=False).sum()),
    "invisible_chars(FEFF/ZW/bidi)": int(txt.str.contains(INVISIBLE).sum()),
    "tatweel": int(txt.str.contains("ـ", regex=False).sum()),
    "diacritics": int(txt.str.contains(ARABIC_DIAC).sum()),
    "latin_letters": int(txt.str.contains(r"[A-Za-z]").sum()),
    "digits": int(txt.str.contains(r"[0-9٠-٩۰-۹]").sum()),
    "punctuation_or_symbols": int(txt.map(lambda t: any(unicodedata.category(c)[0] in "PS" for c in t)).sum()),
    "plus_sign": int(txt.str.contains("+", regex=False).sum()),
}
print(json.dumps(noise, ensure_ascii=False, indent=1))

key = txt.map(compare_norm)
dup_text = key[key.duplicated(keep=False) & (key != "")]
dup_tab = (dup_text.value_counts().rename_axis("norm_text").reset_index(name="n_clips"))
dup_tab["n_words"] = dup_tab["norm_text"].str.split().str.len()
dup_tab.to_csv(OUT / "duplicate_transcripts.csv", index=False, encoding="utf-8-sig")
print(f"repeated transcripts: {len(dup_tab)} distinct texts covering {int(dup_tab['n_clips'].sum())} clips")

def fold_all(w):
    return re.sub(r"[أإآ]", "ا", w).replace("ى", "ي").replace("ة", "ه")
words = collections.Counter(w for t in txt for w in re.sub(r"[^ء-ي\s]", " ", ARABIC_DIAC.sub("", INVISIBLE.sub("", t))).split())
groups = collections.defaultdict(list)
for w, c in words.items():
    groups[fold_all(w)].append((w, c))
var_rows = [(k, " | ".join(f"{w}:{c}" for w, c in sorted(v, key=lambda z: -z[1])), sum(c for _, c in v),
             sum(c for _, c in v) - max(c for _, c in v)) for k, v in groups.items() if len(v) > 1]
var_tab = pd.DataFrame(var_rows, columns=["folded", "variants", "tokens", "minority_tokens"]
                       ).sort_values("tokens", ascending=False)
var_tab.to_csv(OUT / "spelling_variants.csv", index=False, encoding="utf-8-sig")
print(f"vocabulary: {len(words)} types, {sum(words.values())} tokens, "
      f"{sum(1 for c in words.values() if c == 1)} singletons")
print(f"spelling variants: {len(var_tab)} folded types, minority-form tokens = {int(var_tab['minority_tokens'].sum())}")

# --- Stage 5 ---
# 5. Exact audio duplicates (PCM hash)
grp = inv.groupby("sha256")["ID"].apply(list)
dup_audio = grp[grp.str.len() > 1]
inv["duplicate_of"] = ""
for ids_ in dup_audio:
    for i in ids_[1:]:
        inv.loc[inv["ID"] == i, "duplicate_of"] = ids_[0]
pd.DataFrame({"ids": [";".join(v) for v in dup_audio]}).to_csv(OUT / "duplicate_audio.csv", index=False)
print("exact duplicate audio groups:", len(dup_audio), "| extra copies:", int((inv["duplicate_of"] != "").sum()))

# --- Stage 6 ---
# 6. Recording sessions (episodes) from intro/outro phrases
OUTRO = re.compile(r"حلقات جديدة")
CLOSING = re.compile(r"^(و?سلام)( سلام)*$")
INTRO = re.compile(r"في حلقة النهار|أنا أشرف إبراهيم")

norm_txt = inv["text"].map(compare_norm).tolist()
ends = []
for k, t in enumerate(norm_txt):
    if OUTRO.search(t):
        j = k
        while j + 1 < len(norm_txt) and CLOSING.match(norm_txt[j + 1]):
            j += 1
        ends.append(j)
ends = sorted(set(ends))
if not ends or ends[-1] != len(inv) - 1:
    ends.append(len(inv) - 1)
ep_id = np.zeros(len(inv), dtype=int); start = 0
for e_i, end in enumerate(ends):
    ep_id[start:end + 1] = e_i; start = end + 1
inv["episode"] = [f"ep{e:03d}" for e in ep_id]
inv["seg_idx"] = inv.groupby("episode").cumcount()

episodes = inv.groupby("episode").agg(first_id=("ID", "first"), last_id=("ID", "last"),
                                      n_clips=("ID", "size"), minutes=("duration", lambda d: d.sum() / 60),
                                      first_text=("text", "first"), last_text=("text", "last")).reset_index()
intro_pos = inv[inv["text"].map(compare_norm).str.contains(INTRO)].groupby("episode")["seg_idx"].apply(list)
episodes["intro_positions"] = episodes["episode"].map(lambda e: intro_pos.get(e, []))
med = episodes["n_clips"].median()
episodes["suspect_merged"] = episodes.apply(
    lambda r: any(p > 150 for p in r["intro_positions"]) or r["n_clips"] > 1.8 * med, axis=1)
episodes.to_csv(OUT / "episodes.csv", index=False, encoding="utf-8-sig")
print(f"episodes: {len(episodes)} | clips/episode median {med:.0f} | minutes min/median/max "
      f"{episodes['minutes'].min():.1f}/{episodes['minutes'].median():.1f}/{episodes['minutes'].max():.1f}")
print("suspected merged episodes:", episodes["suspect_merged"].sum())

# For each suspect, list the clips around each late intro so the split point can be decided.
CLOSE_HINT = re.compile(r"خلصت كلامي|اشترك|حلقات جديدة|سلام|لو عجبت|التعليقات|اراكم الله|تشوفنا|تشوفونا")
ep_rows = []
for _, r in episodes[episodes["suspect_merged"]].iterrows():
    sub = inv[inv["episode"] == r["episode"]]
    late = [p for p in r["intro_positions"] if p > 150] or [None]
    for p in late:
        window = sub[(sub["seg_idx"] >= (p - 120 if p else 0)) & (sub["seg_idx"] <= (p + 5 if p else 0))]
        closing = window[window["text"].map(compare_norm).str.contains(CLOSE_HINT)]
        suggestion = int(closing["seg_idx"].max()) + 1 if len(closing) else None
        ep_rows.append(dict(episode=r["episode"], n_clips=r["n_clips"], late_intro_seg_idx=p,
                            last_closing_phrase_seg_idx=suggestion - 1 if suggestion else None,
                            suggested_split_seg_idx=suggestion,
                            suggested_split_id=(sub.loc[sub["seg_idx"] == suggestion, "ID"].tolist() or [""])[0],
                            decision="", note=""))
        window[["ID", "seg_idx", "text"]].to_csv(OUT / f"episode_context_{r['episode']}_{p}.csv",
                                                 index=False, encoding="utf-8-sig")
pd.DataFrame(ep_rows, columns=["episode", "n_clips", "late_intro_seg_idx", "last_closing_phrase_seg_idx",
                               "suggested_split_seg_idx", "suggested_split_id", "decision", "note"]
             ).to_csv(OUT / "episode_review.csv", index=False, encoding="utf-8-sig")

# --- Stage 7 ---
# 7. Audio pass B: convert to 16 kHz, 16-bit PCM, mono WAV
# The source clips are peak-normalized to 1.0 and resampling overshoots slightly, so every episode
# gets one small gain computed from its *resampled* peak (measured in pass A). One gain per episode
# keeps the clips' levels relative to each other unchanged, and no sample can clip.
ep_peak = inv.groupby("episode")["peak16"].max()
ep_gain = (CFG["gain_headroom"] / ep_peak).clip(upper=1.0)
inv["gain"] = inv["episode"].map(ep_gain)
print("episodes needing gain < 1:", int((ep_gain < 1).sum()))

def convert(args):
    path, out_path, gain = args
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    y = to16k(mono, sr) * gain
    over = int(np.sum(np.abs(y) > 1.0))
    pcm = np.clip(np.round(y * 32767.0), -32768, 32767).astype(np.int16)
    sf.write(out_path, pcm, CFG["target_sr"], subtype="PCM_16")
    return dict(frames16=len(pcm), overshoot16=over,
                sha256_16k=hashlib.sha256(pcm.astype("<i2").tobytes()).hexdigest())

inv["wav16"] = inv["ID"].map(lambda i: str(WAV16 / f"{i}.wav"))
t0 = time.time()
conv = pmap(convert, list(zip(inv["wav_orig"], inv["wav16"], inv["gain"])))
inv = pd.concat([inv, pd.DataFrame(conv)], axis=1)
inv["duration16"] = inv["frames16"] / CFG["target_sr"]
expected = np.round(inv["frames"] * CFG["target_sr"] / inv["sr"])
inv["frame_mismatch16"] = (inv["frames16"] - expected).abs()
print(f"pass B done in {time.time() - t0:.0f}s | max frame mismatch {inv['frame_mismatch16'].max():.0f} "
      f"| samples clipped after conversion {int(inv['overshoot16'].sum())}")

# Acoustic statistics per episode (recording conditions per session)
ep_ac = inv.groupby("episode").agg(speech_db_median=("active_db", "median"), rms_db_median=("rms_db", "median"),
                                   noise_floor_db_median=("noise_floor_db", "median"),
                                   active_frac_median=("active_frac", "median"), peak16_max=("peak16", "max"),
                                   gain=("gain", "first"), dc_median=("dc", "median"))
episodes = episodes.merge(ep_ac.reset_index(), on="episode", how="left")
episodes.to_csv(OUT / "episodes.csv", index=False, encoding="utf-8-sig")
print(episodes[["speech_db_median", "noise_floor_db_median", "active_frac_median", "gain"]].describe().round(2))

# --- Stage 8 ---
# 8. Rule-based flags (computed for every clip)
logc = np.log(inv["cps"].where(inv["n_chars"] > 0))
med_c = np.nanmedian(logc); mad_c = np.nanmedian(np.abs(logc - med_c)) * 1.4826
lo, hi = np.exp(med_c - CFG["cps_mad_k"] * mad_c), np.exp(med_c + CFG["cps_mad_k"] * mad_c)
print(f"speaking-rate band: {lo:.2f} - {hi:.2f} chars/sec (median {np.exp(med_c):.2f})")

flags = collections.defaultdict(list)
def flag(mask, name):
    for i in inv.loc[mask, "ID"]:
        flags[i].append(name)
flag(inv["read_error"] != "", "read_error")
flag((inv["sr"] != CFG["orig_sr"]) | (inv["channels"] != 1) | (inv["subtype"] != "FLOAT"), "format")
flag(inv["n_nonfinite"] > 0, "nonfinite")
flag(inv["duration"] < CFG["min_dur_s"], "too_short")
flag((inv["cps"] < lo) & (inv["n_words"] > 1), "speed_slow")
flag(inv["cps"] > hi, "speed_fast")
flag(inv["active_frac"] < 0.2, "silence")
flag(inv["n_clip_runs"] > 0, "clipping")
flag(inv["duplicate_of"] != "", "duplicate")
flag(inv["text"].str.strip() == "", "empty_text")
inv["flags"] = inv["ID"].map(lambda i: ";".join(flags.get(i, [])))
print(pd.Series([f for v in flags.values() for f in v]).value_counts())

# --- Stage 9 ---
# 9. Speaker consistency (ECAPA embeddings on the 16 kHz audio)
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    torch, HAS_GPU = None, False
print("GPU:", torch.cuda.get_device_name(0) if HAS_GPU else None)
inv["spk_dist_ep"] = np.nan; inv["spk_dist_global"] = np.nan
if CFG["run_speaker"]:
    os.system("pip -q install speechbrain")
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                         savedir="/kaggle/working/ecapa",
                                         run_opts={"device": "cuda:0" if HAS_GPU else "cpu"})
    embs = np.zeros((len(inv), 192), dtype=np.float32)
    order = np.argsort(inv["frames16"].values)          # batch similar lengths together
    B = 64
    for s in range(0, len(order), B):
        idx = order[s:s + B]
        sigs = [sf.read(inv.at[i, "wav16"], dtype="float32")[0] for i in idx]
        L = max(len(z) for z in sigs)
        batch = torch.zeros(len(sigs), L)
        for k, z in enumerate(sigs):
            batch[k, :len(z)] = torch.from_numpy(z)
        lens = torch.tensor([len(z) / L for z in sigs])
        with torch.no_grad():
            e = enc.encode_batch(batch, lens).squeeze(1).cpu().numpy()
        embs[idx] = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    np.save(OUT / "ecapa_embeddings.npy", embs)
    del enc                                              # free the GPU for Whisper
    if HAS_GPU:
        torch.cuda.empty_cache()
    ok = (inv["duration"] >= CFG["speaker_min_dur_s"]).values
    g = embs[ok].mean(0); g /= np.linalg.norm(g)
    inv["spk_dist_global"] = 1 - embs @ g
    for ep, sub in inv[ok].groupby("episode"):
        c = embs[sub.index].mean(0); c /= np.linalg.norm(c)
        m = (inv["episode"] == ep).values
        inv.loc[m, "spk_dist_ep"] = 1 - embs[m] @ c
    d = inv.loc[ok, "spk_dist_ep"]                      # distance to the clip's own episode centroid
    thr = d.median() + CFG["speaker_mad_k"] * (d - d.median()).abs().median() * 1.4826
    out_mask = ok & (inv["spk_dist_ep"] > thr).values
    flag(out_mask, "speaker")
    print(f"speaker outlier threshold {thr:.3f} | outliers {int(out_mask.sum())}")
    # episode-level check: does any whole episode sound like a different speaker?
    ep_cent = inv[ok].groupby("episode")["spk_dist_global"].median()
    print("episode median distance to global speaker: min %.3f max %.3f" % (ep_cent.min(), ep_cent.max()))
    ep_cent.rename("median_spk_dist").to_csv(OUT / "episode_speaker_distance.csv")
inv["flags"] = inv["ID"].map(lambda i: ";".join(flags.get(i, [])))

# --- Stage 10 ---
# 10. Alignment check with Whisper on the flagged clips (flag only, never auto-exclude)
def cer(ref, hyp):
    r, h = ref.replace(" ", ""), hyp.replace(" ", "")
    if not r:
        return float(len(h) > 0)
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)

inv["whisper_text"] = ""; inv["cer"] = np.nan
if CFG["run_whisper"]:
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=CFG["whisper_model"],
                   torch_dtype=torch.float16 if HAS_GPU else torch.float32,
                   device=0 if HAS_GPU else -1)
    flagged_ids = set(i for i, v in flags.items() if v)
    sel = inv[inv["ID"].isin(flagged_ids)]
    t0 = time.time()
    audio = [{"raw": sf.read(p, dtype="float32")[0], "sampling_rate": CFG["target_sr"]} for p in sel["wav16"]]
    # Greedy decoding, as intended in v1. Recent transformers versions otherwise run beam search for
    # Whisper, which multiplies GPU memory by the number of beams (the cause of the out-of-memory error).
    gen_kwargs = {"language": "arabic", "task": "transcribe", "num_beams": 1}

    def transcribe(items, bs):
        """Transcribe in batches of `bs`; a batch that runs out of GPU memory is retried in halves."""
        out = []
        for s in range(0, len(items), bs):
            chunk = items[s:s + bs]
            try:
                out += asr(chunk, batch_size=len(chunk), generate_kwargs=gen_kwargs)
            except torch.cuda.OutOfMemoryError:
                if HAS_GPU:
                    torch.cuda.empty_cache()
                if len(chunk) == 1:
                    raise
                print(f"  out of GPU memory with batch {len(chunk)}; retrying with {len(chunk) // 2}")
                out += transcribe(chunk, len(chunk) // 2)
        return out

    hyps = transcribe(audio, CFG["whisper_batch"]) if audio else []
    inv.loc[sel.index, "whisper_text"] = [h["text"].strip() for h in hyps]
    inv.loc[sel.index, "cer"] = [cer(compare_norm(r), compare_norm(h["text"])) for r, h in zip(sel["text"], hyps)]
    print(f"Whisper on {len(sel)} flagged clips in {time.time() - t0:.0f}s")
inv["flags"] = inv["ID"].map(lambda i: ";".join(flags.get(i, [])))

# --- Stage 11 ---
# 11. Joins between consecutive clips (for building long sequences)
nxt = inv.groupby("episode").shift(-1)
joins = pd.DataFrame({
    "episode": inv["episode"], "left_id": inv["ID"], "right_id": nxt["ID"],
    "left_tail_db": inv["tail_db"], "right_head_db": nxt["head_db"],
    "left_active_db": inv["active_db"], "right_active_db": nxt["active_db"],
    "left_text": inv["text"], "right_text": nxt["text"],
}).dropna(subset=["right_id"])
# edge level relative to the clip's own speech level: near 0 dB = speech running into the cut
joins["left_tail_rel"] = joins["left_tail_db"] - joins["left_active_db"]
joins["right_head_rel"] = joins["right_head_db"] - joins["right_active_db"]
joins["risk"] = joins[["left_tail_rel", "right_head_rel"]].min(axis=1)
joins["voiced_both_edges"] = (joins["left_tail_rel"] > -10) & (joins["right_head_rel"] > -10)
joins.to_csv(OUT / "joins_all.csv", index=False, encoding="utf-8-sig")
print(f"joins: {len(joins)} | speech at both edges: {int(joins['voiced_both_edges'].sum())} "
      f"({joins['voiced_both_edges'].mean():.1%})")

n = CFG["join_review_n"]
riskiest = joins.sort_values("risk", ascending=False).head(n // 2)
rand = joins.drop(riskiest.index).sample(n - len(riskiest), random_state=CFG["seed"])
jr = pd.concat([riskiest.assign(sample="riskiest"), rand.assign(sample="random")])
JOINDIR = OUT / "join_samples"; JOINDIR.mkdir(exist_ok=True)
for _, r in jr.iterrows():
    a = sf.read(inv.loc[inv["ID"] == r["left_id"], "wav16"].iloc[0], dtype="int16")[0]
    b = sf.read(inv.loc[inv["ID"] == r["right_id"], "wav16"].iloc[0], dtype="int16")[0]
    sf.write(JOINDIR / f"{r['left_id']}_{r['right_id']}.wav", np.concatenate([a, b]), CFG["target_sr"], subtype="PCM_16")
jr["decision"] = ""; jr["note"] = ""
jr.to_csv(OUT / "joins_review.csv", index=False, encoding="utf-8-sig")

# --- Stage 12 ---
# 12. Review table with predefined rule-based suggestions
# Rules are fixed here, before anyone looks at model results, and applied the same way to every clip.
def suggest(r):
    f = set(r["flags"].split(";")) - {""}
    c = r["cer"]
    if f & {"read_error", "nonfinite", "empty_text"}:
        return "exclude", "unreadable or empty"
    if "duplicate" in f:
        return "exclude", f"exact audio copy of {r['duplicate_of']}"
    if r["duration"] < 0.1 and r["n_words"] >= 2:
        return "exclude", "duration cannot contain the text"
    if "silence" in f and (np.isnan(c) or c > CFG["cer_exclude"]):
        return "exclude", "mostly silent and text not recognised"
    if f & {"speed_fast", "speed_slow", "too_short"} and not np.isnan(c) and c > CFG["cer_exclude"]:
        return "exclude", "implausible speaking rate and Whisper disagrees"
    if "speaker" in f and not np.isnan(c) and c > CFG["cer_exclude"]:
        return "exclude", "speaker outlier and Whisper disagrees"
    if f and not np.isnan(c) and c <= 0.3:
        return "keep", "flagged, but Whisper confirms the text"
    return "review", "needs a decision"

cand = inv[inv["flags"] != ""].copy()
sug = cand.apply(suggest, axis=1, result_type="expand")
cand["suggestion"], cand["suggestion_reason"] = (sug[0], sug[1]) if len(cand) else ([], [])
cand["decision"] = ""; cand["note"] = ""
cols = ["ID", "episode", "seg_idx", "flags", "duration", "n_words", "cps", "active_frac", "n_clip_runs", "max_clip_run",
        "spk_dist_ep", "spk_dist_global", "duplicate_of", "text", "whisper_text", "cer", "suggestion", "suggestion_reason",
        "decision", "note"]
cand[cols].to_csv(OUT / "review_candidates.csv", index=False, encoding="utf-8-sig")
print("review candidates:", len(cand))
print(cand["suggestion"].value_counts())

# The "+" in the channel name: does Whisper hear "بلس"?
plus = inv[inv["text"].str.contains("+", regex=False) & (inv["whisper_text"] != "")]
plus_rate = plus["whisper_text"].str.contains("بلس|بلاس|plus", case=False).mean() if len(plus) else np.nan
print(f"'+' clips transcribed by Whisper: {len(plus)} | contain 'بلس': {plus_rate}")

# --- Stage 13 ---
# 13. Save inventory, figures and the audit report
inv_cols = ["ID", "episode", "seg_idx", "text", "wav_orig", "wav16", "sr", "channels", "subtype", "frames",
            "duration", "frames16", "duration16", "frame_mismatch16", "peak", "peak16", "gain", "n_clip_runs",
            "max_clip_run", "overshoot16", "n_nonfinite", "rms_db", "active_db", "active_frac", "noise_floor_db", "head_db", "tail_db",
            "dc", "sha256", "sha256_16k", "duplicate_of", "n_words", "n_chars", "cps", "spk_dist_ep",
            "spk_dist_global", "whisper_text", "cer", "flags", "read_error"]
inv[inv_cols].to_csv(OUT / "inventory.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
ax[0].hist(inv["duration"], bins=60); ax[0].set_title("Clip duration (s)")
ax[1].hist(inv["cps"].clip(upper=40), bins=80); ax[1].axvline(lo, c="r"); ax[1].axvline(hi, c="r")
ax[1].set_title("Speaking rate (chars/s)")
ax[2].hist(episodes["minutes"], bins=30); ax[2].set_title("Episode length (min)")
plt.tight_layout(); plt.savefig(FIG / "distributions.png", dpi=120)

sha = hashlib.sha256(raw_bytes).hexdigest()
report = f"""# EJUST audit report ({CFG['version']}, notebook v2: clipping fix)

Generated {time.strftime('%Y-%m-%d %H:%M')} from `{SEG_CSV}` (sha256 {sha[:16]}…).

## Inventory
- Clips: {len(inv)} | transcripts: {len(seg)} | read errors: {int((inv['read_error'] != '').sum())}
- Formats: {inv[['sr', 'channels', 'subtype']].astype(str).agg(' / '.join, axis=1).value_counts().to_dict()}
- Total audio: {inv['duration'].sum() / 3600:.3f} h
- Duration (s): min {inv['duration'].min():.3f}, median {inv['duration'].median():.2f}, mean {inv['duration'].mean():.2f}, max {inv['duration'].max():.2f}; clips > 10 s: {int((inv['duration'] > 10).sum())}
- Peak: median {inv['peak'].median():.3f}, max {inv['peak'].max():.3f} (files are peak-normalized; a peak of 1.0 is not clipping)
- Clipping (run of >= {CFG['clip_min_run']} samples at |x| >= {CFG['clip_level']}): {int((inv['n_clip_runs'] > 0).sum())} clips; total runs {int(inv['n_clip_runs'].sum())}; longest run {int(inv['max_clip_run'].max())} samples
- Non-finite: {int((inv['n_nonfinite'] > 0).sum())}

## Speakers and sessions
- Speaker metadata: none. ECAPA outliers (distance to episode centroid): {int(inv['flags'].str.contains('speaker').sum())}
- Episodes recovered: {len(episodes)} (clips median {med:.0f}; minutes {episodes['minutes'].min():.1f}–{episodes['minutes'].max():.1f}); suspected merged: {int(episodes['suspect_merged'].sum())}

## Acoustic statistics per episode (`episodes.csv`)
- median speech level (dBFS) across episodes: min {episodes['speech_db_median'].min():.1f}, median {episodes['speech_db_median'].median():.1f}, max {episodes['speech_db_median'].max():.1f}
- median noise floor (dBFS) across episodes: min {episodes['noise_floor_db_median'].min():.1f}, median {episodes['noise_floor_db_median'].median():.1f}, max {episodes['noise_floor_db_median'].max():.1f}
- median speech-active fraction across episodes: min {episodes['active_frac_median'].min():.3f}, max {episodes['active_frac_median'].max():.3f}

## Transcripts
- Noise: {json.dumps(noise, ensure_ascii=False)}
- Vocabulary {len(words)} types / {sum(words.values())} tokens / {sum(1 for c in words.values() if c == 1)} singletons
- Repeated transcripts: {len(dup_tab)} texts, {int(dup_tab['n_clips'].sum())} clips
- Spelling-variant minority tokens (after folding أإآ/ى/ة): {int(var_tab['minority_tokens'].sum())}
- Speaking-rate band: {lo:.2f}–{hi:.2f} chars/s

## Duplicates
- Exact audio duplicate groups: {len(dup_audio)}

## 16 kHz conversion
- Files written: {int(inv['frames16'].notna().sum())}; max frame mismatch {inv['frame_mismatch16'].max():.0f}; samples clipped after conversion {int(inv['overshoot16'].sum())}
- Per-episode gain from the resampled 16 kHz peak (headroom {CFG['gain_headroom']}): {int((ep_gain < 1).sum())} episodes with gain < 1 (min {ep_gain.min():.4f}, median {ep_gain.median():.4f})

## Joins
- Consecutive pairs: {len(joins)}; speech at both edges: {int(joins['voiced_both_edges'].sum())}

## Flags and suggestions
- Flag counts: {pd.Series([f for v in flags.values() for f in v]).value_counts().to_dict()}
- Review candidates: {len(cand)}; suggestions: {cand['suggestion'].value_counts().to_dict()}
- '+' clips with Whisper output: {len(plus)}; share containing "بلس": {plus_rate}
"""
(OUT / "audit_report.md").write_text(report, encoding="utf-8")
write_env_lock()
print(report)

# --- Stage 14 ---
# 14. Validation checks (the notebook fails loudly if any of these break)
assert len(inv) == len(wav_files) == len(seg), "clip / transcript count mismatch"
assert (inv["read_error"] == "").all(), "unreadable files: see inventory.csv"
assert inv["frames16"].notna().all(), "some files were not converted"
assert (inv["frame_mismatch16"] <= 1).all(), "16 kHz length differs from expected"
assert int(inv["overshoot16"].sum()) == 0, "samples clipped during 16 kHz conversion"
assert len(list(WAV16.glob("*.wav"))) == len(inv), "wav16k folder count mismatch"
for p, n16 in zip(inv["wav16"], inv["frames16"]):      # every converted file
    i = sf.info(p)
    assert i.samplerate == CFG["target_sr"] and i.channels == 1 and i.subtype == "PCM_16" and i.frames == n16, p
assert inv["episode"].notna().all() and inv.groupby("episode")["order"].apply(lambda s: s.is_monotonic_increasing).all()
assert (episodes["n_clips"].sum() == len(inv))
print("All validation checks passed.")
print("Outputs in", OUT)
for p in sorted(OUT.iterdir()):
    print(" ", p.name)
