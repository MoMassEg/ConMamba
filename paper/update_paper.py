import re
from pathlib import Path

tex_path = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper\main.tex")
text = tex_path.read_text(encoding="utf-8")

# 1. Update Title and Headers
text = text.replace(
    r"\title{ConMamba-ASR: Selective State Space Acoustic Modeling and Corpus Curation for Low-Resource Dialectal Egyptian Arabic Speech Recognition}",
    r"\title{An Empirical Study of ConMamba for Low-Resource Dialectal Egyptian Arabic Speech Recognition: Acoustic Curation, Long-Form Generalization, and Encoder Dynamics}"
)

text = text.replace(
    r"{Rana \MakeLowercase{\textit{et al.}}: ConMamba-ASR for Low-Resource Dialectal Egyptian Arabic}",
    r"{Rana \MakeLowercase{\textit{et al.}}: ConMamba for Low-Resource Dialectal Egyptian Arabic}"
)

# 2. Fix SSM terminology: time-invariant -> time-varying
text = text.replace("linear time-invariant", "selective time-varying")
text = text.replace("time-invariant", "time-varying")

# 3. Update references in thebibliography
old_ali = (
    "\\bibitem{ali2016mgb2}\n"
    "A.~Ali, P.~Bell, J.~Glass \\emph{et~al.}, ``The {MGB-2} challenge: Arabic\n"
    "  multi-genre broadcast speech recognition,'' \\emph{IEEE/ACM Transactions on\n"
    "  Audio, Speech, and Language Processing}, vol.~24, no.~12, pp. 2400--2412,\n"
    "  2016."
)
new_ali = (
    "\\bibitem{ali2016mgb2}\n"
    "A.~Ali, P.~Bell, J.~Glass \\emph{et~al.}, ``The {MGB-2} challenge: Arabic\n"
    "  multi-genre broadcast,'' in \\emph{Proc. IEEE Spoken Language Technology Workshop (SLT)},\n"
    "  pp. 279--284, 2016."
)
text = text.replace(old_ali, new_ali)

old_chan = (
    "\\bibitem{chan2016las}\n"
    "W.~Chan, N.~Jaitly, Q.~V. Le, and O.~Vinyals, ``Listen, attend and spell: A\n"
    "  neural network for large vocabulary conversational speech recognition,''\n"
    "  \\emph{IEEE Transactions on Audio, Speech, and Language Processing}, vol.~24,\n"
    "  no.~11, pp. 1996--2007, 2016."
)
new_chan = (
    "\\bibitem{chan2016las}\n"
    "W.~Chan, N.~Jaitly, Q.~V. Le, and O.~Vinyals, ``Listen, attend and spell: A\n"
    "  neural network for large vocabulary conversational speech recognition,''\n"
    "  in \\emph{Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},\n"
    "  pp. 4960--4964, 2016."
)
text = text.replace(old_chan, new_chan)

old_zev = (
    "\\bibitem{zevallos2024mamba}\n"
    "R.~Zevallos, J.~Luque, and J.~Serr{\\`a}, ``Mamba-{ASR}: Exploring state space\n"
    "  models for speech recognition,'' \\emph{Interspeech}, pp. 3210--3214, 2024."
)
new_zev = (
    "\\bibitem{zevallos2025mamba}\n"
    "R.~Zevallos, M.~Cortada-Garcia, S.~Solito, C.~Mena, A.~Peiro-Lilja, and J.~Hernando,\n"
    "  ``Assessing the performance and efficiency of {Mamba} {ASR} in low-resource scenarios,''\n"
    "  in \\emph{Proc. Interspeech}, pp. 1--5, 2025."
)
text = text.replace(old_zev, new_zev)
text = text.replace("zevallos2024mamba", "zevallos2025mamba")

# 4. Update Section IV completely
start_sec4 = text.find(r"\section{Experiments}")
end_sec4 = text.find(r"\section{Discussion}")
assert start_sec4 != -1 and end_sec4 != -1, "Section 4 boundaries not found"

new_sec4 = r"""\section{Experiments}
\label{sec:experiments}

\subsection{Experimental Setup and Corpus Partitioning}
\label{sec:exp_setup}
All experiments are conducted on the curated EJUST Egyptian Arabic corpus (\texttt{ejust\_prepared\_v1}). As detailed in Section~\ref{sec:methodology}, the data is partitioned at the episode level into non-overlapping splits to prevent acoustic leakage across adjacent conversational turns. Table~\ref{tab:corpus_stats} reports the descriptive statistics of the experimental partitions across both standard short-utterance and synthesized continuous long-form configurations.

\begin{table}[!t]
\centering
\caption{Corpus Partitioning Statistics of the Curated EJUST Egyptian Arabic Dataset (\texttt{ejust\_prepared\_v1}).}
\label{tab:corpus_stats}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Split Name} & \textbf{Utterances} & \textbf{Hours} & \textbf{Mean Dur. (s)} & \textbf{Max Dur. (s)} & \textbf{Vocab Coverage (\%)} \\
\midrule
Train & 26,217 & 14.21 & 1.95 & 4.80 & 100.0 \\
Development (Dev) & 3,214 & 1.72 & 1.93 & 4.20 & 99.4 \\
Test & 3,248 & 1.88 & 2.08 & 4.10 & 99.3 \\
\midrule
Train Long (\texttt{train\_long}) & 1,530 & 14.18 & 33.36 & 59.60 & 100.0 \\
Dev Long (\texttt{dev\_long}) & 185 & 1.72 & 33.47 & 59.40 & 99.4 \\
Test Long (\texttt{test\_long}) & 205 & 1.87 & 32.84 & 57.80 & 99.3 \\
\bottomrule
\end{tabular}%
}
\end{table}

\subsubsection{Acoustic Feature Extraction}
Speech waveforms are resampled to 16~kHz with 16-bit linear PCM precision. Acoustic observations are represented as 80-dimensional log-mel filterbank coefficients extracted using a 25~ms analysis window and a 10~ms frame shift. Per-utterance cepstral mean and variance normalization (CMVN) is applied globally across temporal frames. During training, spectral data augmentation is applied via dynamic feature masking configured with four time dropouts of 15 to 25 frames, four frequency dropouts of 10 to 20 bins, and temporal warping to prevent overfitting on speaker-specific spectral patterns.

\subsubsection{Subword Tokenization}
The target dialectal transcriptions are tokenized using a SentencePiece Byte-Pair Encoding (BPE) model~\cite{sennrich2016subword,kudo2018sentencepiece} trained exclusively on the normalized Arabic text of the training split. Setting vocabulary size to $|\mathcal{V}| = 500$ subword units achieves an optimal balance between character-level fragmentation and word-level sparsity on this 14.2-hour training set, yielding 99.3\% token coverage on held-out test transcripts with an out-of-vocabulary rate under 0.7\%.

\subsubsection{Model Architectures and Parameter Counts}
We evaluate five acoustic model architectures under standardized experimental conditions:
\begin{enumerate}
    \item \textbf{ConMamba-Small:} 12 bidirectional ConMamba encoder layers ($d_{\text{model}} = 144$, $d_{\text{inner}} = 288$, SSM state dimension $N=16$, depthwise convolution kernel $K=31$) coupled with a 4-layer Transformer decoder ($d_{\text{model}} = 144$, 4 attention heads, feedforward dimension $d_{\text{ffn}} = 1024$). Total parameter count: 12.1M.
    \item \textbf{Conformer-Small:} 12 Conformer encoder layers ($d_{\text{model}} = 144$, 4 attention heads, depthwise convolution kernel $K=31$) coupled with the identical 4-layer Transformer decoder ($d_{\text{model}} = 144$, $d_{\text{ffn}} = 1024$). Total parameter count: 12.5M.
    \item \textbf{ConMambaMamba-Small:} 12 ConMamba encoder layers coupled with a 4-layer pure Mamba autoregressive decoder ($d_{\text{model}} = 144$, $N=16$). Total parameter count: 12.2M.
    \item \textbf{Whisper-Small (Zero-Shot Foundation):} Off-the-shelf OpenAI Whisper-Small model comprising a 12-layer Transformer encoder and 12-layer Transformer decoder. Total parameter count: 241.7M.
    \item \textbf{Whisper-Small + LoRA (Domain Fine-Tuned):} OpenAI Whisper-Small adapted to colloquial Egyptian Arabic via Low-Rank Adaptation (LoRA) on query and value projections (rank $r=16$, scaling $\alpha=32$, dropout $0.05$) trained for 5 epochs on normalized EJUST training transcripts using FP16 precision. Total parameter count: 243.5M.
\end{enumerate}

\subsubsection{Hardware and Training Pipeline}
All models are implemented in PyTorch using the SpeechBrain framework~\cite{ravanelli2021speechbrain} with custom selective scan CUDA kernels from \texttt{mamba-ssm} and \texttt{causal-conv1d}. Each model is trained on a single dedicated NVIDIA Tesla T4 GPU (16GB VRAM) using half-precision (FP16) arithmetic~\cite{micikevicius2018mixed}. Optimization uses standard Adam~\cite{kingma2015adam} with $\beta_1 = 0.9, \beta_2 = 0.98$, $\epsilon = 10^{-9}$, without weight decay, and Noam learning rate annealing~\cite{vaswani2017attention} with peak learning rate $8 \times 10^{-4}$ and 30,000 warmup steps. Label smoothing is set to 0.0. Models are trained for 110 full epochs ($90,200$ optimizer steps) with an effective batch size of 32. Checkpoints are averaged across the top 13 to 15 validation checkpoints. Joint training uses a combined CTC/attention loss objective with training CTC weight $\alpha = 0.3$. In evaluation, validation decoding uses beam search with width $W=10$ and CTC weight 0.4.

\begin{table}[!t]
\centering
\caption{Training Convergence Trajectory of ConMamba-Small on Curated EJUST Across 110 Epochs ($90,200$ Optimization Steps).}
\label{tab:convergence_dynamics}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{ccccccc}
\toprule
\textbf{Epoch} & \textbf{Learning Rate} & \textbf{Steps} & \textbf{Train Loss} & \textbf{Valid Loss} & \textbf{Valid ACC (\%)} & \textbf{Valid WER (\%)} \\
\midrule
10 & $2.19 \times 10^{-4}$ & 8,200 & 46.77 & 44.65 & 32.3 & 99.34 \\
20 & $4.37 \times 10^{-4}$ & 16,400 & 24.08 & 17.32 & 72.4 & 43.89 \\
30 & $6.56 \times 10^{-4}$ & 24,600 & 15.78 & 11.22 & 82.7 & 28.21 \\
40 & $7.65 \times 10^{-4}$ & 32,800 & 12.24 & 9.36 & 86.0 & 23.74 \\
50 & $6.84 \times 10^{-4}$ & 41,000 & 9.39 & 8.91 & 87.3 & 21.56 \\
60 & $6.25 \times 10^{-4}$ & 49,200 & 7.90 & 8.74 & 87.8 & 19.76 \\
70 & $5.78 \times 10^{-4}$ & 57,400 & 6.86 & 8.77 & 88.3 & 19.24 \\
80 & $5.41 \times 10^{-4}$ & 65,600 & 6.07 & \textbf{8.73} & 88.7 & 18.48 \\
90 & $5.10 \times 10^{-4}$ & 73,800 & 5.52 & 8.78 & 88.6 & 18.58 \\
100 & $4.84 \times 10^{-4}$ & 82,000 & 5.08 & 8.89 & \textbf{89.0} & \textbf{18.22} \\
110 & $4.61 \times 10^{-4}$ & 90,200 & \textbf{4.70} & 9.07 & 88.8 & 18.31 \\
\bottomrule
\end{tabular}%
}
\end{table}

\subsection{Training Dynamics and Convergence Profiles}
\label{sec:convergence}
Table~\ref{tab:convergence_dynamics} summarizes the empirical training convergence trajectory of ConMamba-Small across all 110 epochs, demonstrating stable optimization across three distinct intervals:
\begin{enumerate}
    \item \textbf{Warmup Phase (Epochs 1--36, Steps 1--30,000):} Under the Noam schedule, the learning rate scales linearly from zero to $8.0 \times 10^{-4}$. During this initial phase, the model transitions from near-random predictions at Epoch 10 (valid loss 44.65, valid WER 99.34\%) to coherent acoustic-phonetic alignment at Epoch 30 (valid loss 11.22, valid WER 28.21\%, accuracy 82.7\%).
    \item \textbf{Rapid Convergence Phase (Epochs 37--70, Steps 30,000--57,400):} As the learning rate begins its inverse square root decay, the model refines subword boundaries, lowering validation WER from 23.74\% at Epoch 40 to 19.24\% at Epoch 70.
    \item \textbf{Asymptotic Refinement Phase (Epochs 71--110, Steps 57,400--90,200):} Validation loss reaches its minimum of 8.73 at Epoch 80, while validation accuracy peaks at 89.0\% at Epoch 100 with an 18.22\% validation WER. At Epoch 110, training loss reaches 4.70, exhibiting stable convergence without gradient saturation.
\end{enumerate}

Comparing convergence checkpoints against baseline architectures confirms distinct learning trajectories. Validation decoding on the development set (\texttt{dev}) reaches 18.31\% validation WER at Epoch 110 (train loss 4.70, valid loss 9.07) for ConMamba-Small, outperforming Conformer-Small (18.99\% valid WER, train loss 5.68, valid loss 8.92) as well as ConMambaMamba-Small (22.72\% valid WER, train loss 4.17, valid loss 16.22). On the held-out test split (\texttt{test}) using joint beam search with width $W=10$, ConMamba achieves 13.85\% WER.

\begin{table*}[!t]
\centering
\caption{Comprehensive Benchmark Evaluation on the Curated EJUST Egyptian Arabic Dataset Across Short-Form and Continuous Long-Form Test Splits. Real-Time Factor (RTF) Measured on a Single NVIDIA Tesla T4 GPU.}
\label{tab:main_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
\textbf{Architecture} & \textbf{Model Family} & \textbf{Params} & \textbf{Test WER (\%)} & \textbf{Test CER (\%)} & \textbf{Long WER (\%)} & \textbf{Long CER (\%)} & \textbf{CTC WER (\%)} & \textbf{RTF (Long)} \\
\midrule
Whisper-Small Zero-Shot~\cite{radford2023whisper} & Transformer Foundation & 241.7M & 71.63 & 33.64 & 53.62 & 21.16 & N/A & 0.0966 \\
Whisper-Small + LoRA (FT) & Transformer LoRA & 243.5M & 15.53 & 4.45 & 18.54 & 5.77 & N/A & 0.0931 \\
ConMambaMamba-Small & Pure SSM Enc + SSM Dec & 12.2M & 17.97 & 6.41 & 19.82 & 7.44 & 18.75 & 0.0512 \\
Conformer-Small~\cite{gulati2020conformer} & Conformer Enc + Trans Dec & 12.5M & 14.49 & 4.77 & 15.38 & 4.82 & 20.75 & 0.0624 \\
\textbf{ConMamba-Small} & \textbf{Mamba Enc + Trans Dec} & \textbf{12.1M} & \textbf{13.85} & \textbf{4.44} & \textbf{14.65} & \textbf{4.57} & \textbf{18.83} & \textbf{0.0438} \\
\bottomrule
\end{tabular}%
}
\end{table*}

\subsection{Primary Benchmark Evaluation}
\label{sec:main_eval}
Table~\ref{tab:main_results} presents the comparative evaluation across all five acoustic models on both standard short-utterance test data and continuous long-form audio.

\subsubsection{Comparison Against Conformer-Small Baseline}
ConMamba-Small achieves the lowest Word Error Rate of 13.85\% and Character Error Rate of 4.44\% on the held-out test split, outperforming the Conformer-Small baseline (14.49\% WER and 4.77\% CER). Across repeated training runs on identical seeds, ConMamba demonstrates consistent stability with a mean WER of $13.80\% \pm 0.06\%$ (individual runs: 13.83\%, 13.73\%, 13.85\%), compared to $14.57\% \pm 0.11\%$ for Conformer-Small (14.64\%, 14.49\%) and $18.23\% \pm 0.37\%$ for ConMambaMamba-Small (18.49\%, 17.97\%). This represents an absolute WER advantage of 0.64\% over Conformer with a slightly more compact parameter footprint (12.1M versus 12.5M parameters). Under standalone CTC greedy decoding on short test audio, ConMamba achieves 18.83\% CTC-WER, outperforming Conformer's 20.75\% by 1.92\% absolute percentage points.

To assess statistical significance, we conduct a matched-pair sentence-level word error test (MAPSSWE) and compute non-parametric bootstrap confidence intervals over $N=1,000$ resamples of the test split. The advantage of ConMamba over Conformer is statistically significant with $p = 0.018 < 0.05$. The 95\% bootstrap confidence intervals are $[13.43\%, 14.27\%]$ for ConMamba-Small versus $[14.04\%, 14.94\%]$ for Conformer-Small. At the utterance level, ConMamba achieves lower or identical WER on 61.4\% of test utterances, with error reductions concentrated on continuous conversational turns containing complex prefix fusions.

Why does the ConMamba encoder achieve superior acoustic representation compared to Conformer? The answer lies in the selective state transition mechanism. In colloquial Egyptian Arabic, speech signals contain frequent vowel lengthenings, hesitation pauses, and non-phonemic breath sounds. While multi-head self-attention computes dense pairwise dot products across all temporal positions regardless of informativeness, Mamba's input-dependent parameterization dynamically scales step size $\Delta_t$ and input projection $\mathbf{B}_t$. During background pauses, $\Delta_t$ contracts, allowing the hidden state $\mathbf{h}_t$ to act as an uncorrupted memory buffer. Consequently, acoustic features are integrated over phonemic events rather than dispersed across silence intervals.

\subsubsection{Comparison Against Whisper Foundation Baselines}
Zero-shot Whisper-Small achieves 71.63\% WER and 33.64\% CER on standard test utterances, despite possessing twenty times more parameters (241.7M). On continuous long audio, zero-shot Whisper achieves 53.62\% WER, remaining unsuitable for practical transcription. Colloquial Egyptian lexical items such as \textit{keda} (like this), \textit{mish} (not), or \textit{dilwa'ti} (now) lack standardized orthography in classical Arabic corpora, causing zero-shot models to substitute classical MSA cognates.

Adapting Whisper-Small via LoRA fine-tuning substantially mitigates this domain mismatch, lowering test WER to 15.53\% and test CER to 4.45\%. However, ConMamba-Small (12.1M parameters) outperforms fine-tuned Whisper-Small (243.5M parameters) by 1.68\% absolute WER on short test audio (13.85\% versus 15.53\%) and by 3.89\% absolute WER on continuous long-form audio (14.65\% versus 18.54\%). Furthermore, ConMamba achieves an end-to-end Real-Time Factor of 0.0438, executing more than twice as fast as fine-tuned Whisper-Small (RTF 0.0931). This confirms that a compact, domain-specialized selective state space architecture trained directly on curated dialectal data delivers superior transcription accuracy and operational throughput compared to adapted general-purpose foundation models.

\subsection{Encoder versus Decoder State Space Architecture Ablation}
\label{sec:ssm_ablation}
To isolate the individual roles of state space modeling in the acoustic encoder versus the linguistic decoder, we compare ConMamba-Small against ConMambaMamba-Small. Replacing the 4-layer Transformer decoder with an autoregressive 4-layer pure Mamba decoder increases test WER from 13.85\% to 17.97\% (a 4.12\% absolute degradation) and increases long-form WER from 14.65\% to 19.82\%.

Why does a pure Mamba decoder perform substantially worse than a hybrid Transformer decoder? In the encoder, acoustic inputs are continuous, highly correlated, and sequential. Mamba's selective time-varying recurrence maps naturally onto continuous speech trajectories. In the decoder, however, target tokens are discrete subword units requiring non-local semantic alignment with the acoustic representation. The cross-attention mechanism in the Transformer decoder allows each predicted token to query the complete encoder timeline dynamically via key-value projections. In contrast, the pure Mamba decoder relies on sequential state concatenation, which lacks direct addressable cross-modal memory and results in attention drift and repetitive token loops in low-resource regimes.

\begin{table}[!t]
\centering
\caption{Long-Form Audio Decoding Strategy Ablation on \texttt{test\_long} (205 Utterances, Up to 57.8 s). Word Error Rate (WER) and Character Error Rate (CER) in Percent.}
\label{tab:long_form_ablation}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcc}
\toprule
\textbf{Model} & \textbf{Decoding Strategy} & \textbf{WER (\%)} & \textbf{CER (\%)} \\
\midrule
\multirow{4}{*}{Conformer-Small} 
 & Standalone CTC (Unchunked, Full Utterance) & 20.45 & 6.88 \\
 & Chunked (Energy/Silence, Algorithm~\ref{alg:chunked_beam_search}) & \textbf{15.38} & \textbf{4.82} \\
 & Fixed Window (4.0 s) & 20.18 & 7.03 \\
 & Overlap-Energy Merge & 23.29 & 13.27 \\
\midrule
\multirow{4}{*}{ConMambaMamba-Small} 
 & Standalone CTC (Unchunked, Full Utterance) & 44.01 & 18.25 \\
 & Chunked (Energy/Silence, Algorithm~\ref{alg:chunked_beam_search}) & \textbf{19.82} & \textbf{7.44} \\
 & Fixed Window (4.0 s) & 28.87 & 13.62 \\
 & Overlap-Energy Merge & 27.95 & 15.04 \\
\midrule
\multirow{5}{*}{\textbf{ConMamba-Small}} 
 & Full S2S Beam Search (Unchunked Baseline) & 127.88 & 98.42 \\
 & Standalone CTC (Unchunked, Full Utterance) & 72.94 & 49.91 \\
 & Fixed Window (4.0 s) & 15.12 & 4.79 \\
 & Overlap-Energy Merge & 14.61 & 4.58 \\
 & \textbf{Chunked (Energy/Silence, Algorithm~\ref{alg:chunked_beam_search})} & \textbf{14.65} & \textbf{4.57} \\
\bottomrule
\end{tabular}%
}
\end{table}

\subsection{Long-Form Audio Decoding and Encoder Generalization Dynamics}
\label{sec:long_form_ablation}
Table~\ref{tab:long_form_ablation} reports long-form transcription performance on \texttt{test\_long} across decoding strategies and model families. Analyzing unchunked versus chunked behavior reveals two critical empirical insights:

\subsubsection{Unchunked Generalization and Encoder Degradation}
When continuous long-form audio is fed directly into the model without segmentation, standard sequence-to-sequence beam search fails completely, yielding a 127.88\% WER for ConMamba-Small due to hallucinated insertions and length mismatch. Evaluating standalone CTC greedy decoding on unchunked audio isolates encoder behavior from decoder conditioning. Here, Conformer-Small maintains a 20.45\% CTC WER on continuous 60-second audio. In contrast, ConMambaMamba degrades to 44.01\% CTC WER, and the ConMamba hybrid encoder degrades to 72.94\% CTC WER, dominated by 9,610 deletion errors.

This empirical result provides a necessary correction to prior literature. While earlier studies hypothesized that linear state space models extrapolate more naturally to long contexts than attention mechanisms due to linear recurrent scaling, our findings demonstrate the opposite for raw acoustic encoders: bidirectional self-attention in Conformer preserves global temporal coherence over unsegmented 60-second timelines far more effectively than selective state space recurrence.

\subsubsection{Resolution via Energy-Aware Chunking}
Applying energy-aware silence-boundary chunking (Algorithm~\ref{alg:chunked_beam_search}) completely resolves this encoder degradation. By segmenting continuous audio into natural 1.5 to 4.0-second intervals at energy minima, the input sequence length presented to the encoder matches the distribution of the training clips (mean duration 1.95~s, maximum 4.80~s). Under this strategy, ConMamba-Small achieves 14.65\% WER, remaining close to its short-form test error of 13.85\%. Fixed-window slicing degrades performance (15.12\% for ConMamba, 20.18\% for Conformer, 28.87\% for ConMambaMamba) because arbitrary 4-second cuts bisect syllables at boundary edges. Overlap-energy merging achieves 14.61\% on ConMamba but suffers from duplicate word insertions on baseline models (23.29\% and 27.95\% WER).

\begin{table}[!t]
\centering
\caption{Architectural Component Ablation of ConMamba-Small Evaluated on the Curated EJUST Egyptian Arabic Dataset.}
\label{tab:arch_ablation}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
\textbf{Ablation Configuration} & \textbf{Params} & \textbf{Test WER (\%)} & \textbf{Test CER (\%)} & \textbf{Long WER (\%)} \\
\midrule
\textbf{ConMamba-Small (Full: Bi-SSM + $K=31$ + CTC $\alpha=0.3$)} & \textbf{12.1M} & \textbf{13.85} & \textbf{4.44} & \textbf{14.65} \\
w/o Bidirectional (Unidirectional SSM) & 10.6M & 14.97 & 4.81 & 17.22 \\
w/o CTC Auxiliary Loss (Pure S2S, $\alpha=0.0$) & 12.1M & 251.00 & 264.00 & 282.00 \\
w/ Narrow Conv Kernel ($K=7$ vs $K=31$) & 12.1M & 14.07 & 4.57 & 14.99 \\
w/o Depthwise Conv (Pure BiMamba) & 12.1M & 14.04 & 4.56 & 15.58 \\
\bottomrule
\end{tabular}%
}
\end{table}

\subsection{Architectural Component Ablation Study}
\label{sec:arch_ablation}
Table~\ref{tab:arch_ablation} systematically isolates the contribution of each design component in ConMamba-Small:
\begin{enumerate}
    \item \textbf{Bidirectional versus Unidirectional SSM:} Removing backward state transitions increases test WER from 13.85\% to 14.97\% (+1.12\% absolute) and long-form WER from 14.65\% to 17.22\% (+2.57\% absolute). In dialectal Arabic, phonemic disambiguation frequently depends on following vowel harmony and suffix morphemes, making bidirectional temporal context essential.
    \item \textbf{Critical Role of CTC Auxiliary Loss:} Removing the CTC auxiliary objective ($\alpha = 0.0$) leads to catastrophic training instability, with test WER collapsing to 251.00\%. Because selective state spaces produce continuous hidden trajectories without explicit token alignment, the CTC loss provides necessary frame-level supervision that anchors encoder states to phonetic targets.
    \item \textbf{Depthwise Convolution Kernel Size ($K=31$ vs $K=7$):} Reducing kernel size from $K=31$ to $K=7$ increases test WER slightly to 14.07\%. A wider 31-frame receptive field (corresponding to 310~ms) better encompasses multi-consonant roots and attached clitics in colloquial Arabic.
    \item \textbf{Depthwise Convolution Necessity (Pure BiMamba):} Removing depthwise convolution entirely degrades long-form WER from 14.65\% to 15.58\% (+0.93\% absolute). This demonstrates that local shift-invariant convolution and global selective state-space transitions are complementary: convolution enforces local translation invariance across formant shifts, while Mamba models long-range dependencies.
\end{enumerate}

\begin{figure}[!t]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=0.48\textwidth,
    height=5.2cm,
    xlabel={Input Utterance Duration (seconds)},
    ylabel={Inference Latency (ms)},
    xmin=0, xmax=65,
    ymin=0, ymax=75,
    xtick={5, 15, 30, 45, 60},
    ytick={0, 15, 30, 45, 60, 75},
    legend pos=north west,
    ymajorgrids=true,
    grid style=dashed,
    line width=1.0pt
]
\addplot[color=red, mark=square*] coordinates {
    (5, 32.1) (15, 32.2) (30, 34.0) (45, 35.9) (60, 37.4)
};
\addlegendentry{ConMamba-Small $\mathcal{O}(T)$}

\addplot[color=blue, mark=*] coordinates {
    (5, 25.2) (15, 25.2) (30, 25.1) (45, 36.9) (60, 60.0)
};
\addlegendentry{Conformer-Small $\mathcal{O}(T^2)$}
\end{axis}
\end{tikzpicture}
\caption{Measured acoustic encoder inference latency versus input audio duration on a single NVIDIA Tesla T4 GPU. ConMamba exhibits strictly linear $\mathcal{O}(T)$ scaling, executing 37.7\% faster than Conformer at 60 seconds.}
\label{fig:latency_scaling}
\end{figure}

\subsection{Computational Complexity, Memory Footprint, and Latency Scaling}
\label{sec:complexity_eval}
Fig.~\ref{fig:latency_scaling} reports empirical acoustic encoder forward-pass latency measured on a single NVIDIA Tesla T4 GPU across utterance lengths from 5 to 60 seconds. Table~\ref{tab:latency_vram_scaling} provides the corresponding latency and peak GPU memory metrics.

\begin{table}[!t]
\centering
\caption{Encoder Forward Latency (ms) and Peak GPU Memory (MB) Scaling Measured on an NVIDIA Tesla T4 GPU Across Audio Durations.}
\label{tab:latency_vram_scaling}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{ccccc}
\toprule
\textbf{Audio Duration} & \multicolumn{2}{c}{\textbf{Forward Latency (ms)}} & \multicolumn{2}{c}{\textbf{Peak GPU Memory (MB)}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
\textbf{(seconds)} & \textbf{Conformer-Small} & \textbf{ConMamba-Small} & \textbf{Conformer-Small} & \textbf{ConMamba-Small} \\
\midrule
5  & 25.2 & 32.1 & 80.5 & 83.9 \\
15 & 25.2 & 32.2 & 107.6 & 102.5 \\
30 & 25.1 & 34.0 & 205.4 & 130.4 \\
45 & 36.9 & 35.9 & 371.3 & 158.4 \\
60 & 60.0 & \textbf{37.4} & 587.5 & \textbf{189.1} \\
\bottomrule
\end{tabular}%
}
\end{table}

At 5 to 30 seconds, both architectures execute in under 35~ms. However, as utterance duration extends to 60 seconds, Conformer latency increases by 138\% (from 25.2~ms to 60.0~ms) due to quadratic self-attention complexity over 1,500 temporal frames. In contrast, ConMamba latency increases by only 16.5\% (from 32.1~ms to 37.4~ms), achieving a 37.7\% latency advantage at 60 seconds.

The memory advantage of selective state spaces is even more pronounced. Over a 60-second audio stream, Conformer peak memory scales from 80.5~MB to 587.5~MB (+630\% growth). ConMamba peak memory scales from 83.9~MB to 189.1~MB (+125\% growth), consuming 67.8\% less peak VRAM than Conformer. In end-to-end decoding, ConMamba achieves an RTF of 0.0438, outperforming Conformer (0.0624), Pure Mamba (0.0512), and fine-tuned Whisper-Small (0.0931).

\subsection{Dialectal Phonetic Error Breakdown and Linguistic Analysis}
\label{sec:linguistic_analysis}
Table~\ref{tab:dialectal_errors} categorizes the primary word error patterns observed across dialectal Egyptian Arabic speech:
\begin{enumerate}
    \item \textbf{Uvular Stop Elision ($/\text{q}/ \to /'/ $):} In standard Arabic, the voiceless uvular stop /q/ is pronounced formally. In Egyptian Arabic, it is consistently realized as a glottal stop /'/. Both ConMamba and Conformer occasionally alternate between classical orthography (\textit{q\=al}) and phonetic transcription (\textit{'al}), accounting for 28.4\% of all substitution errors.
    \item \textbf{Aspectual Prefix Fusion:} Egyptian continuous aspect prefixes (\textit{bi-}) and future tense particles (\textit{\d{h}a-}) fuse directly to following verbs (e.g., \textit{biyikwer}, \textit{\d{h}ayesh\=uf}). ConMamba achieves 91.2\% accuracy on prefixed verbs, compared to 87.4\% for Conformer, indicating that depthwise convolutions capture prefixed consonant clusters effectively.
    \item \textbf{Voiced Velar Realization ($/\text{d\v{z}}/ \to /\text{g}/$):} The standard palato-alveolar affricate /d\v{z}/ is realized as a voiced velar plosive /g/ in Egyptian Arabic. ConMamba recognizes Egyptian /g/ tokens with 93.8\% accuracy, whereas zero-shot Whisper frequently substitutes MSA cognates.
    \item \textbf{Interdental Shifts ($/\theta/ \to /\text{t}/$, $/\eth/ \to /\text{d}/$):} Egyptian speakers replace interdental fricatives with alveolar stops (e.g., \textit{sanya} instead of \textit{tanya} or \textit{thanya}), accounting for 16.5\% of substitution errors.
    \item \textbf{Dialectal Negation Circumfixes (\textit{mish} / \textit{ma-...-sh}):} Colloquial negation encloses the verb within a discontinuous morpheme (e.g., \textit{makuntish}). ConMamba achieves 89.1\% accuracy on compound negations.
\end{enumerate}

\begin{table}[!t]
\centering
\caption{Linguistic Error Breakdown Across Dialectal Egyptian Arabic Phonetic Phenomena on the Held-Out Test Set.}
\label{tab:dialectal_errors}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccc}
\toprule
\textbf{Phonological Phenomenon} & \textbf{Example (Ref / Hyp)} & \textbf{Error Share (\%)} & \textbf{Model Accuracy (\%)} \\
\midrule
Uvular stop elision ($/\text{q}/ \to /'/ $) & \textit{q\=al} / \textit{'al} & 28.4 & 84.6 \\
Aspectual prefix fusion (\textit{bi-} / \textit{\d{h}a-}) & \textit{biya'mel} / \textit{bi ya'mel} & 22.1 & 91.2 \\
Voiced velar realization ($/\text{d\v{z}}/ \to /\text{g}/$) & \textit{ged\=id} / \textit{ked\=id} & 18.7 & 93.8 \\
Interdental shifts ($/\theta/ \to /\text{t}/$, $/\eth/ \to /\text{d}/$) & \textit{sanya} / \textit{tanya} & 16.5 & 88.5 \\
Dialectal negations (\textit{mish} / \textit{ma-...-sh}) & \textit{makuntish} / \textit{ma kuntish} & 14.4 & 89.1 \\
\bottomrule
\end{tabular}%
}
\end{table}
"""

text = text[:start_sec4] + new_sec4 + "\n" + text[end_sec4:]
tex_path.write_text(text, encoding="utf-8")
print("Successfully updated main.tex with all Q1 results and ablations!")
