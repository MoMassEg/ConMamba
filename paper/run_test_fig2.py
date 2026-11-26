from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
tex_file = paper_dir / "test_fig.tex"
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

tex_code = r"""\documentclass[journal]{IEEEtran}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows.meta,positioning,fit,backgrounds,calc}
\usepackage{amsmath,amssymb}

\begin{document}

\begin{figure*}[!t]
\centering
\begin{tikzpicture}[
    scale=0.86, every node/.style={transform shape},
    font=\sffamily\footnotesize,
    % Box styles
    box/.style={rectangle, draw=#1!75!black, fill=#1!7, line width=0.7pt, rounded corners=3pt, align=center, inner sep=4pt},
    header/.style={rectangle, fill=#1!18, draw=#1!80!black, line width=0.8pt, rounded corners=3pt, font=\sffamily\bfseries\footnotesize, align=center, inner sep=4pt},
    subbox/.style={rectangle, draw=#1!60!black, fill=white, line width=0.5pt, rounded corners=2pt, align=center, inner sep=3pt},
    arrow/.style={-{Stealth[scale=0.9]}, line width=0.8pt, draw=black!75},
    interarrow/.style={-{Stealth[scale=1.05]}, line width=1.1pt, draw=#1!80!black}
]

% =========================================================================
% COLUMN 1: PHASE 1 - ACOUSTIC CURATION PIPELINE
% =========================================================================
\node[header=blue, minimum width=5.2cm] (h1) at (0, 0) {Phase 1: Multi-Stage Corpus Curation};

\node[box=blue, below=0.35cm of h1, minimum width=5.0cm] (p1_raw) {
    \textbf{Raw E-JUST Conversational Corpus}\\
    \scriptsize 32,716 audio clips, 44.1 kHz, 16-bit Float\\
    \scriptsize 76 recording sessions (72 primary + 4 merged)
};

\node[box=blue, below=0.32cm of p1_raw, minimum width=5.0cm] (p1_resamp) {
    \textbf{Sinc Resampling \& Headroom Scaling}\\
    \scriptsize Kaiser-windowed sinc filter $\to$ 16 kHz PCM\\
    \scriptsize Peak headroom attenuation ($g_e \le 0.99$, 0 clipped)
};

\node[box=blue, below=0.32cm of p1_resamp, minimum width=5.0cm] (p1_ecapa) {
    \textbf{Neural Acoustic Audit (ECAPA-TDNN)}\\
    \scriptsize 192-d deep speaker verification embeddings\\
    \scriptsize Cosine distance $d \le 0.603$ (21 corrupted clips purged)
};

\node[box=blue, below=0.32cm of p1_ecapa, minimum width=5.0cm] (p1_splits) {
    \textbf{Curated Benchmark Corpus Splits}\\
    \scriptsize 17.8 total hours (14.2h Train / 1.8h Dev / 1.8h Test)\\
    \scriptsize Continuous Long-Form sequences ($\le 59.6$~s)
};

\draw[arrow] (p1_raw) -- (p1_resamp);
\draw[arrow] (p1_resamp) -- (p1_ecapa);
\draw[arrow] (p1_ecapa) -- (p1_splits);

% =========================================================================
% COLUMN 2: PHASE 2 - CONMAMBA ACOUSTIC BACKBONE
% =========================================================================
\node[header=red, minimum width=5.6cm] (h2) at (5.9, 0) {Phase 2: ConMamba Acoustic Backbone};

\node[box=red, below=0.35cm of h2, minimum width=5.4cm] (p2_fbank) {
    \textbf{Log-Mel Feature Extraction}\\
    \scriptsize 80 Mel filterbanks, 25 ms window, 10 ms hop\\
    \scriptsize SpecAugment (4 time drops + 4 freq drops + time warp)
};

\node[box=red, below=0.32cm of p2_fbank, minimum width=5.4cm] (p2_conv2d) {
    \textbf{2D-CNN Subsampling ($4\times$ Decimation)}\\
    \scriptsize 2 Conv2d layers ($3\times 3$, stride 2), $d_{\text{model}} = 144$
};

% ConMamba Stacked Blocks Box
\node[box=red, fill=red!3, draw=red!80!black, below=0.32cm of p2_conv2d, minimum width=5.4cm, inner sep=4pt] (p2_block) {
    \textbf{Stacked ConMamba Blocks ($\times 12$, 12.1M Params)}\\[3pt]
    \begin{tikzpicture}[every node/.style={transform shape}]
        \node[subbox=red, minimum width=4.9cm] (ssm) {
            \textbf{Bidirectional Selective SSM ($d_{\text{state}}=16$)}\\
            \scriptsize Forward $\mathbf{h}_t^{(f)} \in \mathbb{R}^{16}$ \ \textbf{\&}\ \ Backward $\mathbf{h}_t^{(b)} \in \mathbb{R}^{16}$ ($\mathcal{O}(T)$)
        };
        \node[subbox=red, below=0.18cm of ssm, minimum width=4.9cm] (conv) {
            \textbf{Depthwise Separable Conv ($K=31$)}\\
            \scriptsize Pointwise $\to$ SiLU $\to$ Depthwise ($K=31$) $\to$ Pointwise
        };
        \node[subbox=red, below=0.18cm of conv, minimum width=4.9cm] (res) {
            \scriptsize Pre-LayerNorm, Residual Skip Addition, Dropout $p=0.1$
        };
        \draw[arrow] (ssm) -- (conv);
        \draw[arrow] (conv) -- (res);
    \end{tikzpicture}
};

\draw[arrow] (p2_fbank) -- (p2_conv2d);
\draw[arrow] (p2_conv2d) -- (p2_block);

% Inter-phase arrow 1 -> 2 (Routed via gutter)
\coordinate (g1) at ($(p1_splits.east)!0.5!(p2_fbank.west |- p1_splits.east)$);
\draw[interarrow=blue] (p1_splits.east) -- (g1) -- (g1 |- p2_fbank.west) -- (p2_fbank.west)
    node[pos=0.5, above, font=\scriptsize\bfseries, text=blue!90!black] {16 kHz Audio};

% =========================================================================
% COLUMN 3: PHASE 3 - HYBRID CTC/ATTENTION DECODING
% =========================================================================
\node[header=teal, minimum width=5.6cm] (h3) at (12.1, 0) {Phase 3: Hybrid Decoding \& Inference};

% Target Tokens Input at top of Phase 3
\node[box=teal, below=0.35cm of h3, minimum width=5.4cm] (p3_tgt) {
    \textbf{Target Token Sequence (Training)}\\
    \scriptsize BPE-500 Vocabulary, Causal Right-Shift $\mathbf{y}_{<u}$
};

% Dual Decoding Branches (CTC and Transformer Decoder side-by-side)
\node[box=teal, below=0.32cm of p3_tgt, minimum width=2.55cm] (p3_ctc) at (10.75, -2.15) {
    \textbf{CTC Branch ($\alpha = 0.3$)}\\
    \scriptsize Linear Proj. $144 \to 500$\\
    \scriptsize Frame Loss $\mathcal{L}_{\text{CTC}}$
};

\node[box=teal, below=0.32cm of p3_tgt, minimum width=2.65cm] (p3_dec) at (13.45, -2.15) {
    \textbf{Transformer Decoder}\\
    \scriptsize 4 layers, $d_{\text{ffn}}=1024$\\
    \scriptsize Cross-Attention $\mathcal{L}_{\text{seq}}$
};

% Token flow into Transformer Decoder
\draw[arrow] (p3_tgt.south -| p3_dec.north) -- (p3_dec.north);

% Energy Chunking Engine
\node[box=teal, below=0.35cm of $(p3_ctc.south)!0.5!(p3_dec.south)$, minimum width=5.4cm] (p3_chunk) {
    \textbf{Energy-Aware Chunked Search Engine}\\
    \scriptsize Audio $>4.5$~s split at silence minima (RMS $\le -35$~dB)\\
    \scriptsize Dynamic chunking $1.5-4.0$~s preserves local context
};

% Joint Beam Search
\node[box=teal, fill=teal!12, draw=teal!85!black, below=0.32cm of p3_chunk, minimum width=5.4cm] (p3_beam) {
    \textbf{Joint CTC/Attention Beam Search ($W=10, \gamma=0.4$)}\\
    \scriptsize $\text{Score} = (1-\gamma)\log P_{\text{att}}(\mathbf{y}|\mathbf{x}) + \gamma \log P_{\text{ctc}}(\mathbf{y}|\mathbf{x})$
};

% Final Output
\node[box=teal, fill=teal!22, draw=teal!90!black, below=0.32cm of p3_beam, minimum width=5.4cm] (p3_out) {
    \textbf{Recognized Egyptian Arabic Transcription}\\
    \scriptsize Dialectal Orthography (500 SentencePiece Subword Units)
};

% Column 3 Internal Arrows
\draw[arrow] (p3_ctc.south) -- ++(0,-0.16) -| ($(p3_chunk.north)+(-1.3,0)$);
\draw[arrow] (p3_dec.south) -- ++(0,-0.16) -| ($(p3_chunk.north)+(1.3,0)$);
\draw[arrow] (p3_chunk) -- (p3_beam);
\draw[arrow] (p3_beam) -- (p3_out);

% Inter-phase arrow 2 -> 3 (Encoder representations into CTC & Decoder)
\coordinate (g2) at ($(p2_block.east)!0.4!(p3_ctc.west |- p2_block.east)$);
\draw[interarrow=red] (p2_block.east) -- (g2) -- (g2 |- p3_ctc.west) -- (p3_ctc.west)
    node[pos=0.4, above, font=\scriptsize\bfseries, text=red!90!black] {$\mathbf{h}_{\text{enc}}$};
\draw[interarrow=red] (g2 |- p3_ctc.west) -- ++(0,0) -| (p3_dec.west);

% =========================================================================
% BACKGROUND CONTAINERS
% =========================================================================
\begin{scope}[on background layer]
    \node[draw=blue!50!black, fill=blue!2, rounded corners=5pt, fit=(h1)(p1_raw)(p1_splits), inner sep=6pt] (c1) {};
    \node[draw=red!50!black, fill=red!1, rounded corners=5pt, fit=(h2)(p2_fbank)(p2_block), inner sep=6pt] (c2) {};
    \node[draw=teal!50!black, fill=teal!2, rounded corners=5pt, fit=(h3)(p3_tgt)(p3_out), inner sep=6pt] (c3) {};
\end{scope}

\end{tikzpicture}
\caption{System overview of the proposed empirical framework for dialectal Egyptian Arabic ASR. Phase 1 demonstrates the multi-stage acoustic curation pipeline on the EJUST corpus, resolving resampling-induced digital clipping via per-episode headroom attenuation ($g_e \le 0.99$), removing speaker outliers via ECAPA-TDNN neural embeddings, and building paired short-form and continuous long-form benchmarks ($<60$~s). Phase 2 depicts the hybrid acoustic backbone combining 2D-CNN subsampling ($4\times$), 12 stacked bidirectional ConMamba encoder blocks (with forward/backward selective SSMs and wide depthwise separable convolutions $K=31$), and LayerNorm residuals. Phase 3 illustrates the hybrid CTC/attention decoding topology, coupling a frame-level CTC projection ($\alpha=0.3$) with a 4-layer Transformer cross-attention decoder and energy-aware chunked beam search ($W=10, \gamma=0.4$) for robust long-form transcription over a 500-unit SentencePiece vocabulary.}
\label{fig:framework_overview}
\end{figure*}

\end{document}
"""

with open(tex_file, "w", encoding="utf-8") as f:
    f.write(tex_code)

res = subprocess.run([str(tectonic_exe), str(tex_file)], capture_output=True, text=True, cwd=str(paper_dir))
print("Tectonic exit code:", res.returncode)
if res.returncode != 0:
    print(res.stderr[:800])
else:
    print("Success!")
