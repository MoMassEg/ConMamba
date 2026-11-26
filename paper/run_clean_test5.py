from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
test_tex = paper_dir / "test_fig_clean.tex"
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

tex_code = r"""\documentclass[journal]{IEEEtran}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows.meta,positioning,fit,backgrounds,calc}
\usepackage{amsmath,amssymb}

\begin{document}

\begin{figure*}[!t]
\centering
\begin{tikzpicture}[
    scale=0.88, every node/.style={transform shape},
    font=\sffamily\footnotesize,
    % Strict node geometry: text width=4.7cm, inner sep=3.5pt -> node width strictly 4.95cm
    box/.style={rectangle, draw=#1!75!black, fill=#1!7, line width=0.7pt, rounded corners=3pt, align=center, text width=4.7cm, inner sep=3.5pt},
    header/.style={rectangle, fill=#1!18, draw=#1!80!black, line width=0.8pt, rounded corners=3pt, font=\sffamily\bfseries\footnotesize, align=center, text width=4.7cm, inner sep=3.5pt},
    blockframe/.style={rectangle, draw=#1!80!black, fill=#1!3, dashed, line width=0.8pt, rounded corners=4pt},
    arrow/.style={-{Stealth[scale=0.9]}, line width=0.8pt, draw=black!75},
    interarrow/.style={-{Stealth[scale=1.05]}, line width=1.1pt, draw=#1!80!black}
]

% =========================================================================
% COLUMN 1 (Center x = 0.0): PHASE 1 - ACOUSTIC CURATION PIPELINE
% =========================================================================
\node[header=blue] (h1) at (0, 0) {Phase 1: Multi-Stage Corpus Curation};

\node[box=blue, below=0.35cm of h1] (p1_raw) {
    \textbf{Raw E-JUST Conversational Corpus}\\
    \scriptsize 32,716 audio clips, 44.1 kHz, 16-bit Float\\
    \scriptsize 76 recording sessions (72 primary + 4 merged)
};

\node[box=blue, below=0.32cm of p1_raw] (p1_resamp) {
    \textbf{Sinc Resampling \& Headroom Scaling}\\
    \scriptsize Kaiser sinc filter $\to$ 16 kHz PCM conversion\\
    \scriptsize Episode headroom attenuation ($g_e \le 0.99$, 0 clipped)
};

\node[box=blue, below=0.32cm of p1_resamp] (p1_ecapa) {
    \textbf{Neural Acoustic Audit (ECAPA-TDNN)}\\
    \scriptsize 192-d deep speaker verification embeddings\\
    \scriptsize Cosine distance $d \le 0.603$ (21 corrupted clips purged)
};

\node[box=blue, below=0.32cm of p1_ecapa] (p1_splits) {
    \textbf{Curated Benchmark Splits \& Sequences}\\
    \scriptsize 17.8 total hours (14.2h Train / 1.8h Dev / 1.8h Test)\\
    \scriptsize Continuous Long-Form sequences ($\le 59.6$~s)
};

\draw[arrow] (p1_raw) -- (p1_resamp);
\draw[arrow] (p1_resamp) -- (p1_ecapa);
\draw[arrow] (p1_ecapa) -- (p1_splits);

% =========================================================================
% COLUMN 2 (Center x = 6.2): PHASE 2 - CONMAMBA ACOUSTIC BACKBONE
% =========================================================================
\node[header=red] (h2) at (6.2, 0) {Phase 2: ConMamba Acoustic Backbone};

\node[box=red, below=0.35cm of h2] (p2_fbank) {
    \textbf{Log-Mel Feature Extraction}\\
    \scriptsize 80 Mel filterbanks, 25 ms window, 10 ms hop\\
    \scriptsize SpecAugment (4 time drops + 4 freq drops + time warp)
};

\node[box=red, below=0.28cm of p2_fbank] (p2_conv2d) {
    \textbf{2D-CNN Subsampling ($4\times$ Decimation)}\\
    \scriptsize 2 Conv2d layers ($3\times 3$, stride 2), $d_{\text{model}} = 144$
};

% ConMamba Stacked Blocks (Internal Layers)
\node[box=red, fill=white, below=0.48cm of p2_conv2d, text width=4.3cm] (p2_ssm) {
    \textbf{Bidirectional Selective SSM ($d_{\text{state}}=16$)}\\
    \scriptsize Forward $\mathbf{h}_t^{(f)} \in \mathbb{R}^{16}$ \ \textbf{\&}\ \ Backward $\mathbf{h}_t^{(b)} \in \mathbb{R}^{16}$ ($\mathcal{O}(T)$)
};

\node[box=red, fill=white, below=0.18cm of p2_ssm, text width=4.3cm] (p2_conv) {
    \textbf{Depthwise Separable Conv ($K=31$)}\\
    \scriptsize Pointwise $\to$ SiLU $\to$ Depthwise ($K=31$) $\to$ Pointwise
};

\node[box=red, fill=white, below=0.18cm of p2_conv, text width=4.3cm] (p2_res) {
    \textbf{Feed-Forward \& Residual Normalization}\\
    \scriptsize Pre-LayerNorm, Skip Connection, Dropout $p=0.1$
};

\draw[arrow] (p2_fbank) -- (p2_conv2d);
\draw[arrow] (p2_conv2d) -- (p2_ssm);
\draw[arrow] (p2_ssm) -- (p2_conv);
\draw[arrow] (p2_conv) -- (p2_res);

% Enclosing frame around ConMamba layers
\begin{scope}[on background layer]
    \node[blockframe=red, fit=(p2_ssm)(p2_res), inner sep=5pt,
          label={[red!85!black, font=\sffamily\bfseries\scriptsize]above:Stacked ConMamba Blocks ($\times 12$, 12.1M Params)}] (p2_blockframe) {};
\end{scope}

% =========================================================================
% COLUMN 3 (Center x = 12.4): PHASE 3 - HYBRID DECODING & INFERENCE
% =========================================================================
\node[header=teal] (h3) at (12.4, 0) {Phase 3: Hybrid Decoding \& Inference};

\node[box=teal, below=0.35cm of h3] (p3_dec) {
    \textbf{Transformer Decoder (4 Layers)}\\
    \scriptsize $d_{\text{model}}=144,\ d_{\text{ffn}}=1024,\ 4\text{ heads}$\\
    \scriptsize Cross-attention over acoustic frames $\mathbf{h}_{\text{enc}}$
};

\node[box=teal, below=0.28cm of p3_dec] (p3_ctc) {
    \textbf{CTC Auxiliary Branch ($\alpha = 0.3$)}\\
    \scriptsize Linear Projector $144 \to 500$, Loss $\mathcal{L}_{\text{CTC}}$\\
    \scriptsize Monotonic frame-level phonetic alignment
};

\node[box=teal, below=0.28cm of p3_ctc] (p3_chunk) {
    \textbf{Energy-Aware Chunked Search Engine}\\
    \scriptsize Audio $>4.5$~s split at silence minima (RMS $\le -35$~dB)\\
    \scriptsize Dynamic $1.5-4.0$~s chunks reset recurrent drift
};

\node[box=teal, fill=teal!12, draw=teal!85!black, below=0.28cm of p3_chunk] (p3_beam) {
    \textbf{Joint CTC/Attention Beam Search ($W=10, \gamma=0.4$)}\\
    \scriptsize $\text{Score} = (1-\gamma)\log P_{\text{att}}(\mathbf{y}|\mathbf{x}) + \gamma \log P_{\text{ctc}}(\mathbf{y}|\mathbf{x})$
};

\node[box=teal, fill=teal!22, draw=teal!90!black, below=0.28cm of p3_beam] (p3_out) {
    \textbf{Recognized Egyptian Arabic Text Output}\\
    \scriptsize Dialectal Orthography (500 SentencePiece Units)
};

% Column 3 Internal Arrows
\draw[arrow] (p3_dec) -- (p3_ctc);
\draw[arrow] (p3_ctc) -- (p3_chunk);
\draw[arrow] (p3_chunk) -- (p3_beam);
\draw[arrow] (p3_beam) -- (p3_out);

% =========================================================================
% INTER-PHASE CONNECTIONS THROUGH CLEAR GUTTERS
% =========================================================================
% Gutter 1: fixed at x = 3.10 cm (completely clear space between x=2.45 and x=3.65)
\draw[interarrow=blue] (p1_splits.east) -- (3.10, 0 |- p1_splits.east) -- (3.10, 0 |- p2_fbank.west) -- (p2_fbank.west)
    node[pos=0.5, above, font=\scriptsize\bfseries, text=blue!90!black] {16 kHz Audio};

% Gutter 2: fixed at x = 9.30 cm (completely clear space between x=8.65 and x=9.85)
% Encoder representations h_enc into Transformer Decoder
\draw[interarrow=red] (p2_blockframe.east |- p2_ssm.east) -- (9.30, 0 |- p2_ssm.east) -- (9.30, 0 |- p3_dec.west) -- (p3_dec.west)
    node[pos=0.5, above, font=\scriptsize\bfseries, text=red!90!black] {$\mathbf{h}_{\text{enc}}$ Cross-Attention};

% Encoder representations h_enc into CTC Projector
\draw[interarrow=red] (9.30, 0 |- p2_ssm.east) -- (9.30, 0 |- p3_ctc.west) -- (p3_ctc.west)
    node[pos=0.5, above, font=\scriptsize\bfseries, text=red!90!black] {$\mathbf{h}_{\text{enc}}$ CTC Proj.};

% =========================================================================
% BACKGROUND CONTAINERS (EQUALIZED HEIGHT)
% =========================================================================
\path let \p1 = (p1_splits.south), \p2 = (p2_blockframe.south), \p3 = (p3_out.south) in
    \pgfextra{
        \typeout{SOUTH_BALANCED: p1=\y1, p2=\y2, p3=\y3}
    };

\coordinate (common_bottom) at (0, -290pt);

\begin{scope}[on background layer]
    \node[draw=blue!50!black, fill=blue!2, rounded corners=5pt, fit=(h1)(p1_raw)(p1_splits)(h1 |- common_bottom), inner sep=6pt] (c1) {};
    \node[draw=red!50!black, fill=red!1, rounded corners=5pt, fit=(h2)(p2_fbank)(p2_blockframe)(h2 |- common_bottom), inner sep=6pt] (c2) {};
    \node[draw=teal!50!black, fill=teal!2, rounded corners=5pt, fit=(h3)(p3_dec)(p3_out)(h3 |- common_bottom), inner sep=6pt] (c3) {};
\end{scope}

\end{tikzpicture}
\caption{System overview of the proposed empirical framework for dialectal Egyptian Arabic ASR. Phase 1 demonstrates the multi-stage acoustic curation pipeline on the EJUST corpus, resolving resampling-induced digital clipping via per-episode headroom attenuation ($g_e \le 0.99$), removing speaker outliers via ECAPA-TDNN neural embeddings, and building paired short-form and continuous long-form benchmarks ($<60$~s). Phase 2 depicts the hybrid acoustic backbone combining 2D-CNN subsampling ($4\times$), 12 stacked bidirectional ConMamba encoder blocks (with forward/backward selective SSMs and wide depthwise separable convolutions $K=31$), and LayerNorm residuals. Phase 3 illustrates the hybrid CTC/attention decoding topology, coupling a frame-level CTC projection ($\alpha=0.3$) with a 4-layer Transformer cross-attention decoder and energy-aware chunked beam search ($W=10, \gamma=0.4$) for robust long-form transcription over a 500-unit SentencePiece vocabulary.}
\label{fig:framework_overview}
\end{figure*}

\end{document}
"""

with open(test_tex, "w", encoding="utf-8") as f:
    f.write(tex_code)

res = subprocess.run([str(tectonic_exe), "--print", str(test_tex)], capture_output=True, text=True, cwd=str(paper_dir))
for line in res.stdout.splitlines():
    if "SOUTH_BALANCED" in line:
        print(line)

# Render to PNG
from pdf_to_png import save_pdf_page_as_png
save_pdf_page_as_png(paper_dir / "test_fig_clean.pdf", 0, paper_dir / "test_fig_clean.png", scale=2)
