import os, sys, json, time, logging
from pathlib import Path
import torch
import soundfile as sf
import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

torch.backends.cudnn.benchmark = True
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import if_main_process

logger = logging.getLogger(__name__)

# --- Long-Form Segmentation Strategies (Q1 Option 2) ---
def segment_fixed(wav, sr=16000, max_dur=4.0):
    max_s = int(max_dur * sr)
    return [wav[i:i + max_s] for i in range(0, wav.size(-1), max_s)]

def segment_energy(wav, sr=16000, max_dur=4.0, min_dur=1.5, overlap_dur=0.0, hop_size=160, frame_size=400):
    total = wav.size(-1)
    max_s, min_s, ov_s = int(max_dur * sr), int(min_dur * sr), int(overlap_dur * sr)
    if total <= max_s or total < frame_size:
        return [wav]
    unfolded = wav.unfold(-1, frame_size, hop_size)
    energy = torch.sqrt(torch.mean(unfolded ** 2, dim=-1) + 1e-12)
    chunks, start = [], 0
    while start < total:
        if total - start <= max_s:
            chunks.append(wav[start:])
            break
        f_start = (start + min_s) // hop_size
        f_end = min((start + max_s) // hop_size, energy.size(-1))
        if f_end <= f_start:
            split = min(start + max_s, total)
        else:
            split = (f_start + torch.argmin(energy[f_start:f_end]).item()) * hop_size
        if split <= start:
            split = min(start + max_s, total)
        chunks.append(wav[start:split])
        next_start = max(split - ov_s, start + min_s // 2)
        start = next_start if ov_s > 0 else split
    return chunks

def merge_overlapping_texts(texts, max_ov_words=5):
    """Deduplicate repeated boundary words across overlapping chunks."""
    if not texts:
        return ""
    merged = texts[0].split()
    for txt in texts[1:]:
        words = txt.split()
        if not words:
            continue
        best_k = 0
        limit = min(max_ov_words, len(merged), len(words))
        for k in range(limit, 0, -1):
            if merged[-k:] == words[:k]:
                best_k = k
                break
        merged.extend(words[best_k:])
    return " ".join(merged)

class ASR(sb.core.Brain):
    def _decode_chunks_batched(self, chunks, current_epoch, searcher, batch_size=16):
        """Fast batched chunk decoding on GPU (4-6x faster than sequential loop)."""
        valid_chunks = [c for c in chunks if c.size(0) >= 400]
        if not valid_chunks:
            return []
        texts = []
        for i in range(0, len(valid_chunks), batch_size):
            sub = valid_chunks[i:i + batch_size]
            max_l = max(c.size(0) for c in sub)
            padded = torch.zeros(len(sub), max_l, device=self.device)
            lens = torch.zeros(len(sub), device=self.device)
            for j, c in enumerate(sub):
                padded[j, :c.size(0)] = c
                lens[j] = c.size(0) / max_l
            feats = self.hparams.compute_features(padded)
            feats = self.modules.normalize(feats, lens, epoch=current_epoch)
            src = self.modules.CNN(feats)
            if hasattr(self.modules.Transformer, "encode"):
                enc = self.modules.Transformer.encode(src, wav_len=lens)
            else:
                enc, _ = self.modules.Transformer(src, None, lens)
            hyps, _, _, _ = searcher(enc.detach(), lens)
            for h in hyps:
                t = self.tokenizer.decode_ids(h).strip()
                if t:
                    texts.append(t)
        return texts

    def decode_long_utterance(self, wav, current_epoch, searcher, strategy):
        sr = self.hparams.sample_rate
        if strategy == "fixed":
            chunks = segment_fixed(wav, sr=sr, max_dur=4.0)
            texts = self._decode_chunks_batched(chunks, current_epoch, searcher)
            full_text = " ".join(texts).strip()
        elif strategy == "overlap_energy":
            chunks = segment_energy(wav, sr=sr, max_dur=4.0, min_dur=1.5, overlap_dur=0.5)
            texts = self._decode_chunks_batched(chunks, current_epoch, searcher)
            full_text = merge_overlapping_texts(texts)
        else:
            chunks = segment_energy(wav, sr=sr, max_dur=4.0, min_dur=1.5, overlap_dur=0.0)
            texts = self._decode_chunks_batched(chunks, current_epoch, searcher)
            full_text = " ".join(texts).strip()
        return self.tokenizer.encode_as_ids(full_text)

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        if stage == sb.Stage.TEST:
            self._test_audio_sec += sum(
                wav_lens[k].item() * wavs.size(1) / self.hparams.sample_rate
                for k in range(wavs.size(0))
            )

        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            feats, fea_lens = self.hparams.fea_augment(feats, wav_lens)
            tokens_bos = self.hparams.fea_augment.replicate_labels(tokens_bos)
        src = self.modules.CNN(feats)
        enc_out, pred = self.modules.Transformer(src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index)
        p_ctc = self.hparams.log_softmax(self.modules.ctc_lin(enc_out))
        p_seq = self.hparams.log_softmax(self.modules.seq_lin(pred))

        hyps = None
        is_valid_search = stage == sb.Stage.VALID and (
            current_epoch % self.hparams.valid_search_interval == 0
            or current_epoch == self.hparams.number_of_epochs
        )
        if is_valid_search:
            hyps, _, _, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
        elif stage == sb.Stage.TEST:
            searcher = self.hparams.test_search
            strategy = getattr(self.hparams, "active_long_strategy", "chunked")
            sr = self.hparams.sample_rate
            max_dur = max(wav_lens[k].item() * wavs.size(1) / sr for k in range(wavs.size(0)))
            if max_dur > 4.5 and strategy in ("fixed", "chunked", "overlap_energy"):
                hyps = []
                for b in range(wavs.size(0)):
                    b_samp = int(round(wav_lens[b].item() * wavs.size(1)))
                    b_wav = wavs[b, :b_samp]
                    hyps.append(self.decode_long_utterance(b_wav, current_epoch, searcher, strategy))
            else:
                hyps, _, _, _ = searcher(enc_out.detach(), wav_lens)
        return p_ctc, p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        p_ctc, p_seq, wav_lens, hyps = predictions
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            rep = self.hparams.fea_augment.replicate_labels
            tokens, tokens_lens = rep(tokens), rep(tokens_lens)
            tokens_eos, tokens_eos_lens = rep(tokens_eos), rep(tokens_eos_lens)
        loss_seq = self.hparams.seq_cost(p_seq, tokens_eos, length=tokens_eos_lens).sum()
        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens).sum()
        loss = self.hparams.ctc_weight * loss_ctc + (1 - self.hparams.ctc_weight) * loss_seq
        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            do_wer = (
                stage == sb.Stage.TEST
                or current_epoch % self.hparams.valid_search_interval == 0
                or current_epoch == self.hparams.number_of_epochs
            )
            if do_wer and hyps is not None:
                predicted_words = [self.tokenizer.decode_ids(h).split(" ") for h in hyps]
                target_words = [w.split(" ") for w in batch.wrd]
                self.wer_metric.append(ids, predicted_words, target_words)
                self.cer_metric.append(ids, predicted_words, target_words)
                if stage == sb.Stage.TEST and hasattr(self, "ctc_wer_metric"):
                    ctc_words = []
                    argmax_toks = torch.argmax(p_ctc, dim=-1)
                    for b in range(p_ctc.size(0)):
                        v_len = int(round(wav_lens[b].item() * p_ctc.size(1)))
                        seq = argmax_toks[b, :v_len].tolist()
                        collapsed = [k for j, k in enumerate(seq) if j == 0 or k != seq[j - 1]]
                        final_toks = [k for k in collapsed if k != self.hparams.blank_index]
                        c_text = self.tokenizer.decode_ids(final_toks).strip()
                        ctc_words.append(c_text.split(" ") if c_text else [])
                    self.ctc_wer_metric.append(ids, ctc_words, target_words)
                    self.ctc_cer_metric.append(ids, ctc_words, target_words)
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss

    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start()
        ckpts = self.checkpointer.find_checkpoints(max_key=max_key, min_key=min_key)
        ckpt = sb.utils.checkpoints.average_checkpoints(ckpts, recoverable_name="model")
        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()

    def on_stage_start(self, stage, epoch):
        self._t0 = time.time()
        self._test_audio_sec = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()
            self.cer_metric = self.hparams.error_rate_computer(split_tokens=True)
            if stage == sb.Stage.TEST:
                self.ctc_wer_metric = self.hparams.error_rate_computer()
                self.ctc_cer_metric = self.hparams.error_rate_computer(split_tokens=True)

    def _log_efficiency(self, stage, epoch):
        if not if_main_process():
            return
        elapsed = round(time.time() - self._t0, 2)
        rtf = round(elapsed / self._test_audio_sec, 4) if (stage == sb.Stage.TEST and self._test_audio_sec > 0) else None
        rec = {
            "stage": str(stage).split(".")[-1],
            "epoch": epoch,
            "seconds": elapsed,
            "audio_sec": round(self._test_audio_sec, 1) if stage == sb.Stage.TEST else None,
            "rtf": rtf,
            "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
            "test_name": getattr(self.hparams, "test_name", None) if stage == sb.Stage.TEST else None,
        }
        with open(os.path.join(self.hparams.output_folder, "efficiency.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def on_stage_end(self, stage, stage_loss, epoch):
        self._log_efficiency(stage, epoch)
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            return
        stage_stats["ACC"] = self.acc_metric.summarize()
        current_epoch = self.hparams.epoch_counter.current
        do_wer = (
            stage == sb.Stage.TEST
            or current_epoch % self.hparams.valid_search_interval == 0
            or current_epoch == self.hparams.number_of_epochs
        )
        if do_wer:
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            if stage == sb.Stage.TEST and hasattr(self, "ctc_wer_metric"):
                stage_stats["CTC_WER"] = self.ctc_wer_metric.summarize("error_rate")
                stage_stats["CTC_CER"] = self.ctc_cer_metric.summarize("error_rate")
        if stage == sb.Stage.VALID:
            lr = self.hparams.noam_annealing.current_lr
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr, "steps": self.optimizer_step},
                train_stats=self.train_stats, valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"], num_to_keep=self.hparams.avg_checkpoints,
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch": self.hparams.epoch_counter.current, "test": self.hparams.test_name},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w", encoding="utf-8") as w:
                    self.wer_metric.write_stats(w)
                with open(self.hparams.test_wer_file.replace("wer_", "cer_"), "w", encoding="utf-8") as w:
                    self.cer_metric.write_stats(w)
                with open(self.hparams.test_wer_file.replace("wer_", "wer_ctc_"), "w", encoding="utf-8") as w:
                    self.ctc_wer_metric.write_stats(w)
                with open(self.hparams.test_wer_file.replace("wer_", "cer_ctc_"), "w", encoding="utf-8") as w:
                    self.ctc_cer_metric.write_stats(w)

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        if should_step:
            self.hparams.noam_annealing(self.optimizer)

def dataio_prepare(hparams, tokenizer):
    data_folder = hparams["data_folder"]
    load = lambda p: sb.dataio.dataset.DynamicItemDataset.from_csv(csv_path=p, replacements={"data_root": data_folder})
    train_data = load(hparams["train_csv"])
    valid_data = load(hparams["valid_csv"]).filtered_sorted(sort_key="duration")
    test_datasets = {Path(p).stem: load(p).filtered_sorted(sort_key="duration") for p in hparams["test_csv"]}
    datasets = [train_data, valid_data] + list(test_datasets.values())
    valtest = [valid_data] + list(test_datasets.values())

    def read(wav):
        x, sr = sf.read(wav, dtype="float32", always_2d=False)
        return torch.from_numpy(x)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        return read(wav)
    sb.dataio.dataset.add_dynamic_item(valtest, audio_pipeline)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline_train(wav):
        sig = read(wav)
        if "speed_perturb" in hparams:
            sig = hparams["speed_perturb"](sig.unsqueeze(0)).squeeze(0)
        return sig
    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline_train)

    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides("wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens")
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        yield torch.LongTensor([hparams["bos_index"]] + tokens_list)
        yield torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield torch.LongTensor(tokens_list)
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"])
    return train_data, valid_data, test_datasets

if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    sb.utils.distributed.ddp_init_group(run_opts)
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file, overrides=overrides,
    )
    tokenizer = hparams["tokenizer"]
    tokenizer.load(hparams["bpe_model"])
    train_data, valid_data, test_datasets = dataio_prepare(hparams, tokenizer)
    hparams["test_search"] = hparams["valid_search"]
    hparams["test_search"].beam_size = hparams["test_beam_size"]

    asr_brain = ASR(
        modules=hparams["modules"], opt_class=hparams["Adam"],
        hparams=hparams, run_opts=run_opts, checkpointer=hparams["checkpointer"],
    )
    asr_brain.tokenizer = tokenizer

    n_params = sum(p.numel() for p in asr_brain.hparams.model.parameters())
    Path(hparams["output_folder"], "params.json").write_text(json.dumps({"params": n_params}))

    if not hparams["skip_train"]:
        asr_brain.fit(
            asr_brain.hparams.epoch_counter, train_data, valid_data,
            train_loader_kwargs=hparams["train_dataloader_opts"],
            valid_loader_kwargs=hparams["valid_dataloader_opts"],
        )

    os.makedirs(hparams["output_wer_folder"], exist_ok=True)
    strategies = [s.strip() for s in str(hparams.get("long_strategies", "overlap_energy")).split(",")]

    for name, data in test_datasets.items():
        if "long" in name:
            for strat in strategies:
                asr_brain.hparams.active_long_strategy = strat
                asr_brain.hparams.test_name = f"{name}_{strat}"
                asr_brain.hparams.test_wer_file = os.path.join(hparams["output_wer_folder"], f"wer_{name}_{strat}.txt")
                test_opts = dict(hparams["test_dataloader_opts"])
                test_opts["batch_size"] = 1
                asr_brain.evaluate(data, max_key="ACC", test_loader_kwargs=test_opts)
        else:
            asr_brain.hparams.active_long_strategy = "standard"
            asr_brain.hparams.test_name = name
            asr_brain.hparams.test_wer_file = os.path.join(hparams["output_wer_folder"], f"wer_{name}.txt")
            asr_brain.evaluate(data, max_key="ACC", test_loader_kwargs=dict(hparams["test_dataloader_opts"]))
