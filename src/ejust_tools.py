from pathlib import Path
import re

SHIM = """# --- compatibility shim (EJUST) ---
try:
    from causal_conv1d.cpp_functions import (
        causal_conv1d_fwd_function as _cc_new_fwd,
        causal_conv1d_bwd_function as _cc_new_bwd,
    )
    def _cc_fwd(x, weight, bias, seq_idx, silu):
        return _cc_new_fwd(x, weight, bias, seq_idx, None, None, silu)
    def _cc_bwd(x, weight, bias, dout, seq_idx, dx, silu):
        dx, dweight, dbias, _ = _cc_new_bwd(x, weight, bias, dout, seq_idx, None, None, dx, False, silu)
        return dx, dweight, dbias
except ImportError:
    import causal_conv1d_cuda
    def _cc_fwd(x, weight, bias, seq_idx, silu):
        try:
            return causal_conv1d_cuda.causal_conv1d_fwd(x, weight, bias, seq_idx, None, None, silu)
        except TypeError:
            return causal_conv1d_cuda.causal_conv1d_fwd(x, weight, bias, seq_idx, silu)
    def _cc_bwd(x, weight, bias, dout, seq_idx, dx, silu):
        try:
            r = causal_conv1d_cuda.causal_conv1d_bwd(x, weight, bias, dout, seq_idx, None, None, dx, False, silu)
            return r[0], r[1], r[2]
        except TypeError:
            return causal_conv1d_cuda.causal_conv1d_bwd(x, weight, bias, dout, seq_idx, dx, silu)
"""

LN_OLD = "    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn\n"
LN_NEW = (
    "    try:\n"
    "        from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn\n"
    "    except ImportError:\n"
    "        from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn\n"
)

def patch_repo(repo):
    repo = Path(repo)
    p = repo / "modules/mamba/selective_scan_interface.py"
    s = p.read_text()
    if "_cc_fwd" not in s:
        s = re.sub(r"causal_conv1d_cuda\.causal_conv1d_fwd\((x, conv1d_weight)", r"_cc_fwd(\1", s)
        s = s.replace("causal_conv1d_cuda.causal_conv1d_bwd(\n", "_cc_bwd(\n")
        s = s.replace("import causal_conv1d_cuda\n", SHIM, 1)
        p.write_text(s)
    for f in ["modules/mamba/bimamba.py", "modules/mamba/mamba_blocks.py"]:
        p = repo / f
        s = p.read_text()
        if "except ImportError:" not in s and LN_OLD in s:
            p.write_text(s.replace(LN_OLD, LN_NEW, 1))
    p_tasr = repo / "modules/TransformerASR.py"
    if p_tasr.exists():
        s_tasr = p_tasr.read_text()
        old_tgt = "if self.num_decoder_layers > 0:\n            tgt = self.custom_tgt_module(tgt)"
        new_tgt = "if self.num_decoder_layers > 0 and tgt is not None:\n            tgt = self.custom_tgt_module(tgt)"
        if old_tgt in s_tasr:
            p_tasr.write_text(s_tasr.replace(old_tgt, new_tgt, 1))
    return True

def make_hparams(template, out_path, *, data_folder, csv_dir, bpe_model, output_folder,
                 epochs, warmup, valid_search_interval, batch_size=32, lr=0.0008,
                 num_workers=4, grad_accumulation_factor=1, long_strategies="overlap_energy",
                 skip_train=False):
    s = Path(template).read_text()
    rep = [
        (r"^data_folder: !PLACEHOLDER.*$",
         f"data_folder: {data_folder}\nbpe_model: {bpe_model}\nlong_strategies: {long_strategies}"),
        (r"^skip_train: .*$", f"skip_train: {skip_train}"),
        (r"^output_folder: .*$", f"output_folder: {output_folder}"),
        (r"^train_csv: .*$", f"train_csv: {csv_dir}/train.csv"),
        (r"^valid_csv: .*$", f"valid_csv: {csv_dir}/dev.csv"),
        (r"^test_csv:\n(    - .*\n)+",
         f"test_csv:\n    - {csv_dir}/test.csv\n    - {csv_dir}/test_long.csv\n"),
        (r"^number_of_epochs: .*$", f"number_of_epochs: {epochs}"),
        (r"^grad_accumulation_factor: .*$", f"grad_accumulation_factor: {grad_accumulation_factor}"),
        (r"^num_workers: .*$", f"num_workers: {num_workers}"),
        (r"^precision: .*$", "precision: fp16"),
        (r"^lr_adam: .*$", f"lr_adam: {lr}"),
        (r"^batch_size: .*$", f"batch_size: {batch_size}"),
        (r"^dynamic_batching: .*$", "dynamic_batching: False"),
        (r"^valid_dataloader_opts:\n    batch_size: 1$", "valid_dataloader_opts:\n    batch_size: 32"),
        (r"^test_dataloader_opts:\n    batch_size: 1$", "test_dataloader_opts:\n    batch_size: 16"),
        (r"^output_neurons: .*$", "output_neurons: 500"),
        (r"^valid_search_interval: .*$", f"valid_search_interval: {valid_search_interval}"),
        (r"^test_beam_size: .*$", "test_beam_size: 10"),
        (r"^    n_warmup_steps: .*$", f"    n_warmup_steps: {warmup}"),
    ]
    for pat, new in rep:
        s, n = re.subn(pat, new, s, count=1, flags=re.M)
        assert n == 1, f"{Path(template).name}: pattern not found: {pat}"
    Path(out_path).write_text(s)
    return out_path
