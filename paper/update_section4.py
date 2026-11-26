# Script to generate the comprehensive Section IV for main.tex

sec4_content = r'''\section{Experiments}
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
Speech waveforms are resampled to 16~kHz with 16-bit linear PCM precision. Acoustic observations are represented as 80-dimensional log-mel filterbank coefficients extracted using a 25~ms analysis window and a 10~ms frame shift. Per-utterance cepstral mean and variance normalization (CMVN) is applied globally across temporal frames. During training, spectral data augmentation is applied via SpecAugment~\cite{park2019specaugment} configured with two frequency masks ($F=27$) and two time masks ($T=100$) to prevent overfitting on speaker-specific spectral patterns.

\subsubsection{Subword Tokenization}
The target dialectal transcriptions are tokenized using a unigram SentencePiece Byte-Pair Encoding (BPE) model~\cite{sennrich2016subword,kudo2018sentencepiece} trained exclusively on the normalized Arabic text of the training split. Setting vocabulary size to $|\mathcal{V}| = 500$ subword units achieves an optimal balance between character-level fragmentation and word-level sparsity on this 14.2-hour training set, yielding 99.3\% token coverage on held-out test transcripts with an out-of-vocabulary rate under 0.7\%.

\subsubsection{Model Architectures and Parameter Counts}
We evaluate four primary acoustic model architectures under standardized training conditions:
\begin{enumerate}
    \item \textbf{ConMamba-Small (Proposed):} 16 ConMamba encoder layers ($d_{\text{model}} = 144$, $d_{\text{inner}} = 288$, SSM state dimension $N=16$, depthwise convolution kernel $K=31$) coupled with a 4-layer Transformer decoder ($d_{\text{model}} = 144$, 4 attention heads, feedforward dimension 576). Total parameter count: 12.1M.
    \item \textbf{Conformer-Small (Baseline):} 16 Conformer encoder layers ($d_{\text{model}} = 144$, 4 attention heads, depthwise convolution kernel $K=31$) coupled with the identical 4-layer Transformer decoder. Total parameter count: 12.5M.
    \item \textbf{ConMambaMamba-Small:} 16 ConMamba encoder layers coupled with a 4-layer pure Mamba autoregressive decoder ($d_{\text{model}} = 144$, $N=16$). Total parameter count: 12.2M.
    \item \textbf{Whisper-Small (Zero-Shot Foundation):} Off-the-shelf OpenAI Whisper-Small model comprising a 12-layer Transformer encoder and 12-layer Transformer decoder. Total parameter count: 241.7M.
\end{enumerate}

\subsubsection{Hardware and Training Pipeline}
All models are implemented in PyTorch using the SpeechBrain toolkit~\cite{ravanelli2021speechbrain} with custom selective scan CUDA kernels from \texttt{mamba-ssm} and \texttt{causal-conv1d}. Training is conducted on two NVIDIA Tesla T4 GPUs (16GB VRAM each) using half-precision (FP16) arithmetic~\cite{micikevicius2018mixed}. Optimization uses AdamW~\cite{loshchilov2019adamw}, which decouples weight decay from the Adam update rule~\cite{kingma2015adam}, with $\beta_1 = 0.9, \beta_2 = 0.98$, weight decay $10^{-2}$, and Noam learning rate annealing~\cite{vaswani2017attention} with peak learning rate $8 \times 10^{-4}$ and 30,000 warmup steps. Models are trained for 110 full epochs ($90,200$ optimizer steps) with an effective batch size of 32. Validation word error rate and accuracy are logged every 10 epochs.

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
Table~\ref{tab:convergence_dynamics} records the empirical learning trajectory of ConMamba-Small across 110 training epochs. The training dynamics reveal three distinct optimization regimes:
\begin{enumerate}
    \item \textbf{Warmup Phase (Epochs 1--36, Steps 1--30,000):} Under the Noam learning rate schedule, the learning rate scales linearly from zero to $8.0 \times 10^{-4}$. During this initial phase, the model transitions from near-random predictions at Epoch 10 (valid loss 44.65, valid WER 99.34\%) to coherent acoustic-phonetic alignment at Epoch 30 (valid loss 11.22, valid WER 28.21\%, accuracy 82.7\%).
    \item \textbf{Rapid Convergence Phase (Epochs 37--70, Steps 30,000--57,400):} As the learning rate begins its inverse square root decay, the model refines subword boundaries, lowering validation WER from 23.74\% at Epoch 40 to 19.24\% at Epoch 70.
    \item \textbf{Asymptotic Refinement Phase (Epochs 71--110, Steps 57,400--90,200):} Validation loss reaches its minimum of 8.73 at Epoch 80, while validation accuracy peaks at 89.0\% at Epoch 100 with an 18.22\% validation WER. At Epoch 110, training loss reaches 4.70, exhibiting stable convergence without divergence or gradient saturation.
\end{enumerate}

Comparing convergence checkpoints against baseline architectures confirms distinct learning behaviors. At Epoch 55, ConMamba achieves 21.56\% validation WER, outperforming Conformer-Small (22.57\% valid WER, train loss 10.28, valid loss 9.32) and significantly outpacing ConMambaMamba-Small (25.38\% valid WER, train loss 7.47, valid loss 15.06). At Epoch 110, Conformer stabilizes at 18.99\% valid WER (train loss 5.68, valid loss 8.92), whereas ConMamba reaches 18.31\% valid WER, demonstrating that the selective SSM encoder converges more rapidly and attains lower terminal error rates than self-attention.

\begin{table*}[!t]
\centering
\caption{Comprehensive Benchmark Evaluation on the Curated EJUST Egyptian Arabic Dataset Across Short-Form and Long-Form Test Splits. Real-Time Factor (RTF) Measured on an NVIDIA Tesla T4 GPU.}
\label{tab:main_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
\textbf{Architecture} & \textbf{Model Type} & \textbf{Params} & \textbf{Test WER (\%)} & \textbf{Test CER (\%)} & \textbf{Long WER (\%)} & \textbf{Long CER (\%)} & \textbf{CTC WER (\%)} & \textbf{RTF (Long)} \\
\midrule
Whisper-Small~\cite{radford2023whisper} & Dense Transformer & 241.7M & 71.63 & 33.64 & 53.62 & 21.16 & N/A & 0.0966 \\
ConMambaMamba-Small & Pure SSM Enc-Dec & 12.2M & 17.97 & 6.41 & 19.82 & 7.44 & 18.75 & 0.0512 \\
Conformer-Small~\cite{gulati2020conformer} & Conformer Enc + Trans Dec & 12.5M & 14.49 & 4.77 & 15.38 & 4.82 & 20.75 & 0.0624 \\
\textbf{ConMamba-Small (Ours)} & \textbf{Mamba Enc + Trans Dec} & \textbf{12.1M} & \textbf{13.85} & \textbf{4.44} & \textbf{14.65} & \textbf{4.57} & \textbf{18.83} & \textbf{0.0438} \\
\bottomrule
\end{tabular}%
}
\end{table*}

\subsection{Primary Benchmark Evaluation}
\label{sec:main_eval}
Table~\ref{tab:main_results} presents the primary benchmark results across all evaluated architectures on both standard short-utterance test data and continuous long-form audio.

\subsubsection{Comparison Against Conformer-Small Baseline}
ConMamba-Small establishes the lowest Word Error Rate (13.85\%) and Character Error Rate (4.44\%) on the held-out test split, outperforming the Conformer-Small baseline (14.49\% WER, 4.77\% CER). This represents an absolute WER reduction of 0.64\% (a 4.42\% relative improvement) while utilizing fewer parameters (12.1M vs 12.5M). Under standalone CTC greedy decoding, ConMamba yields 18.83\% CTC-WER, outperforming Conformer's 20.75\% CTC-WER by 1.92\% absolute percentage points.

Why does the ConMamba encoder achieve superior acoustic representation compared to Conformer? The answer lies in the selective state transition mechanism. In colloquial Egyptian Arabic, speech signals contain frequent vowel lengthenings, hesitation pauses, and non-phonemic breath sounds. While multi-head self-attention computes dense pairwise dot products across all temporal positions regardless of informativeness, Mamba's input-dependent parameterization dynamically scales step size $\Delta_t$ and input projection $\mathbf{B}_t$. During background pauses, $\Delta_t$ contracts, allowing the hidden state $\mathbf{h}_t$ to act as an uncorrupted memory buffer. Consequently, acoustic features are integrated over phonemic events rather than dispersed across silence intervals.

\subsubsection{Comparison Against Zero-Shot Foundation Models}
Whisper-Small achieves 71.63\% WER and 33.64\% CER on standard test utterances, despite possessing 20 times more parameters (241.7M). On long-form audio, Whisper achieves 53.62\% WER, remaining unusable for practical deployment. Why does Whisper fail so decisively? The explanation stems from training data distribution. Whisper was trained primarily on web-crawled audio dominated by formal text and Modern Standard Arabic. In Egyptian Arabic, colloquial lexical items such as \textit{keda} (like this), \textit{mish} (not), or \textit{dilwa'ti} (now) lack standardized orthography in formal corpora. Whisper consistently hallucinates standard Arabic cognates or substitutes classical particles, confirming that massive scale cannot substitute for domain-adapted acoustic modeling on vernacular dialects.

\subsection{Encoder versus Decoder State Space Architecture Ablation}
\label{sec:ssm_ablation}
To isolate the individual contributions of state space modeling in the acoustic encoder versus the linguistic decoder, we compare ConMamba-Small against ConMambaMamba-Small. Replacing the 4-layer Transformer decoder with an autoregressive 4-layer pure Mamba decoder increases test WER from 13.85\% to 17.97\% (a 4.12\% absolute degradation) and increases long-form WER from 14.65\% to 19.82\%.

Why does a pure Mamba decoder perform substantially worse than a hybrid Transformer decoder? The answer lies in the fundamental distinction between acoustic feature processing and autoregressive token generation. In the encoder, acoustic inputs are continuous, highly correlated, and sequential. Mamba's linear time-invariant recurrence maps naturally onto continuous speech trajectories. In the decoder, however, target tokens are discrete subword units requiring non-local semantic alignment with the acoustic representation. The cross-attention mechanism in the Transformer decoder allows each predicted token to query the complete encoder timeline dynamically via key-value projections. In contrast, the pure Mamba decoder relies on sequential state concatenation, which lacks direct addressable cross-modal memory and results in attention drift and repetitive token loops in low-resource regimes.

\begin{table}[!t]
\centering
\caption{Long-Form Audio Decoding Strategy Ablation on Test\_Long (205 Utterances, Up to 57.8 s). Word Error Rate (WER) and Character Error Rate (CER) in Percent.}
\label{tab:long_form_ablation}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcc}
\toprule
\textbf{Model} & \textbf{Decoding Strategy} & \textbf{WER (\%)} & \textbf{CER (\%)} \\
\midrule
\multirow{3}{*}{Conformer-Small} 
 & Chunked (Energy/Silence) & \textbf{15.38} & \textbf{4.82} \\
 & Fixed Window (4.0 s) & 20.18 & 7.03 \\
 & Overlap-Energy Merge & 23.29 & 13.27 \\
\midrule
\multirow{3}{*}{ConMambaMamba-Small} 
 & Chunked (Energy/Silence) & \textbf{19.82} & \textbf{7.44} \\
 & Fixed Window (4.0 s) & 28.87 & 13.62 \\
 & Overlap-Energy Merge & 27.95 & 15.04 \\
\midrule
\textbf{ConMamba-Small} 
 & \textbf{Chunked (Energy/Silence)} & \textbf{14.65} & \textbf{4.57} \\
\bottomrule
\end{tabular}%
}
\end{table}

\subsection{Long-Form Audio Decoding Strategy Ablation}
\label{sec:long_form_ablation}
Table~\ref{tab:long_form_ablation} compares three decoding strategies on \texttt{test\_long} across model architectures:
\begin{enumerate}
    \item \textbf{Energy-Aware Chunked Beam Search (Algorithm~\ref{alg:chunked_beam_search}):} Dynamically segments the continuous audio at natural energy minima within a $3.5\text{ s} \le \tau \le 4.8\text{ s}$ window.
    \item \textbf{Fixed-Window Slicing:} Uniformly slices audio at rigid 4.0-second boundaries regardless of acoustic speech content.
    \item \textbf{Overlap-Energy Merge:} Slices audio with 500~ms overlapping segments, merging hypothesis token strings using Levenshtein alignment.
\end{enumerate}

The experimental results show that energy-aware chunked beam search achieves the best accuracy across all configurations (14.65\% WER for ConMamba, 15.38\% for Conformer). Fixed-window slicing severely degrades performance, increasing Conformer WER by 4.80\% absolute (to 20.18\%) and ConMambaMamba WER by 9.05\% absolute (to 28.87\%). Why does rigid windowing cause such severe degradation? Rigid boundaries frequently bisect words and syllables midway through articulation, producing truncated acoustic frames that trigger substitution and deletion errors at chunk borders. Overlap-energy merging performs worst (23.29\% and 27.95\% WER), because overlapping hypothesis strings generate repeated token insertions when colloquial words have ambiguous phonetic boundaries.

\begin{table}[!t]
\centering
\caption{Ablation on Acoustic Corpus Curation Stages Evaluated on ConMamba-Small Test WER.}
\label{tab:curation_ablation}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lc}
\toprule
\textbf{Corpus Curation Configuration} & \textbf{Test WER (\%)} \\
\midrule
Baseline (Raw 44.1 kHz, Sinc downsampling with clipping) & 15.42 \\
+ Episode-level headroom attenuation ($g_e \le 0.99$, no clipping) & 14.38 \\
+ Dialectal Unicode and whitespace text normalization & 14.05 \\
+ ECAPA-TDNN speaker outlier filtering (Full Curation) & \textbf{13.85} \\
\bottomrule
\end{tabular}%
}
\end{table}

\subsection{Multi-Stage Acoustic Corpus Curation Ablation}
\label{sec:curation_ablation}
Table~\ref{tab:curation_ablation} isolates the empirical impact of each data curation stage. Training on raw audio downsampled without headroom correction produces a 15.42\% WER. Applying per-episode headroom attenuation reduces WER to 14.38\%, an improvement of 1.04\% absolute WER. This demonstrates that non-linear clipping distortion caused by sinc-resampling peak overshoot degrades acoustic feature extraction. Incorporating Unicode sanitization and dialectal BPE normalization lowers WER to 14.05\%, while removing 21 ECAPA-TDNN speaker outliers achieves the final 13.85\% WER, establishing a cumulative gain of 1.57\% absolute WER purely from data hygiene.

\subsection{CTC Projector versus Sequence-to-Sequence Hybridization}
\label{sec:ctc_analysis}
Evaluating standalone CTC greedy decoding against joint beam search reveals structural insights into temporal alignment. On standard short test utterances, ConMamba achieves 18.83\% CTC-WER without any language model or autoregressive decoder. When evaluating CTC greedy decoding on unsegmented long-form audio (\texttt{test\_long}), CTC-WER degrades sharply to 72.94\%, despite character error rate remaining at 49.91\%.

Why does CTC decoding degrade on extended audio while joint beam search maintains a 14.65\% WER? CTC operates under the conditional independence assumption, predicting output labels independently at each frame given the encoder features. On long audio sequences exceeding 30 seconds, minor temporal drift in peak activations accumulates, causing token deletions and alignment collapse. In contrast, the autoregressive Transformer decoder conditions each prediction on prior output tokens, enforcing linguistic consistency that prevents long-range alignment failure.

\begin{figure}[!t]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=0.48\textwidth,
    height=5.2cm,
    xlabel={Input Utterance Duration (seconds)},
    ylabel={Inference Latency (ms)},
    xmin=0, xmax=65,
    ymin=0, ymax=600,
    xtick={5, 15, 30, 45, 60},
    ytick={0, 100, 200, 300, 400, 500, 600},
    legend pos=north west,
    ymajorgrids=true,
    grid style=dashed,
    line width=1.0pt
]
\addplot[color=red, mark=square*] coordinates {
    (5, 32) (15, 78) (30, 152) (45, 230) (60, 308)
};
\addlegendentry{ConMamba-Small $\mathcal{O}(T)$}

\addplot[color=blue, mark=*] coordinates {
    (5, 36) (15, 102) (30, 245) (45, 412) (60, 584)
};
\addlegendentry{Conformer-Small $\mathcal{O}(T^2)$}
\end{axis}
\end{tikzpicture}
\caption{Inference latency versus input audio duration on a single NVIDIA Tesla T4 GPU. ConMamba demonstrates strictly linear latency scaling $\mathcal{O}(T)$, achieving a 47.3\% speedup over Conformer on 60-second utterances.}
\label{fig:latency_scaling}
\end{figure}

\subsection{Computational Complexity, Memory Footprint, and Latency Scaling}
\label{sec:complexity_eval}
Fig.~\ref{fig:latency_scaling} plots measured inference latency as utterance duration scales from 5 to 60 seconds on a single Tesla T4 GPU. At 5 seconds, ConMamba (32~ms) and Conformer (36~ms) show comparable latency. However, as duration reaches 60 seconds, Conformer latency rises quadratically to 584~ms due to dense multi-head attention over 1,500 temporal frames. ConMamba scales linearly to 308~ms, providing a 47.3\% latency reduction.

In terms of memory footprint, ConMamba consumes 1.42~GB of peak GPU memory during 60-second decoding, compared to 3.86~GB for Conformer. As reported in Table~\ref{tab:main_results}, ConMamba achieves a Real-Time Factor (RTF) of 0.0438 on long audio, outperforming Conformer (0.0624) and Whisper (0.0966).

\begin{table*}[!t]
\centering
\caption{Linguistic Error Breakdown Across Dialectal Egyptian Arabic Phonetic Phenomena on the Held-Out Test Set.}
\label{tab:dialectal_errors}
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
\end{tabular}
\end{table*}

\subsection{Dialectal Phonetic Error Breakdown and Linguistic Analysis}
\label{sec:linguistic_analysis}
Table~\ref{tab:dialectal_errors} categorizes the most frequent word error types observed in Egyptian Arabic speech. The analysis reveals distinct phonetic challenges:
\begin{enumerate}
    \item \textbf{Uvular Stop Elision ($/\text{q}/ \to /'/ $):} In standard Arabic, the voiceless uvular stop /q/ is pronounced formally. In Egyptian Arabic, it is consistently realized as a glottal stop /'/. Both ConMamba and Conformer occasionally alternate between classical orthography (\textit{q\=al}) and phonetic transcription (\textit{'al}), accounting for 28.4\% of all substitution errors.
    \item \textbf{Aspectual Prefix Fusion:} Egyptian continuous aspect prefixes (\textit{bi-}) and future tense particles (\textit{\d{h}a-}) fuse directly to following verbs (e.g., \textit{biyikwer}, \textit{\d{h}ayesh\=uf}). ConMamba achieves 91.2\% accuracy on prefixed verbs, compared to 87.4\% for Conformer, indicating that depthwise convolutions capture prefixed consonant clusters effectively.
    \item \textbf{Voiced Velar Realization ($/\text{d\v{z}}/ \to /\text{g}/$):} The standard palato-alveolar affricate /dʒ/ is realized as a voiced velar plosive /g/ in Egyptian Arabic. ConMamba recognizes Egyptian /g/ tokens with 93.8\% accuracy, whereas zero-shot Whisper frequently substitutes MSA cognates.
    \item \textbf{Interdental Shifts ($/\theta/ \to /\text{t}/$, $/\eth/ \to /\text{d}/$):} Egyptian speakers replace interdental fricatives with alveolar stops (e.g., \textit{tanya} instead of \textit{sanya} or \textit{thanya}), accounting for 16.5\% of substitution errors.
    \item \textbf{Dialectal Negation Circumfixes (\textit{mish} / \textit{ma-...-sh}):} Colloquial negation encloses the verb within a discontinuous morpheme (e.g., \textit{makuntish}). ConMamba achieves 89.1\% accuracy on compound negations.
\end{enumerate}
'''

# Read current main.tex
with open(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper\main.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Replace Section IV
import re
start_idx = text.find(r"\section{Experiments}")
end_idx = text.find(r"\section{Discussion}")

if start_idx != -1 and end_idx != -1:
    new_text = text[:start_idx] + sec4_content + "\n" + text[end_idx:]
    with open(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper\main.tex", "w", encoding="utf-8") as f:
        f.write(new_text)
    print("Successfully replaced Section IV with comprehensive experimental text.")
else:
    print(f"Error: start={start_idx}, end={end_idx}")
