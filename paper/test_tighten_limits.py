from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
main_tex = paper_dir / "main.tex"
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

with open(main_tex, "r", encoding="utf-8") as f:
    text = f.read()

# Remove probe lines
text = text.replace(r"\typeout{PAGE_PROBE_CONCL_START}" + "\n", "")
text = text.replace(r"\typeout{PAGE_PROBE_BIB_START}" + "\n", "")

# In Limitations:
old_item = r"\begin{itemize}"
new_item = r"""\begin{itemize}
\setlength{\itemsep}{2pt}
\setlength{\parskip}{0pt}
\setlength{\parsep}{0pt}"""

# Replace only the one in Limitations
# Let's find \section{Limitations}\n\label{sec:limitations}\nWe identify several limitations in the current study that should guide subsequent research:\n\begin{itemize}
target = r"""\section{Limitations}
\label{sec:limitations}
We identify several limitations in the current study that should guide subsequent research:
\begin{itemize}"""

replacement = r"""\section{Limitations}
\label{sec:limitations}
We identify several limitations in the current study that should guide subsequent research:
\begin{itemize}
\setlength{\itemsep}{1.5pt}
\setlength{\parskip}{0pt}
\setlength{\parsep}{0pt}"""

if target in text:
    text = text.replace(target, replacement)
    print("Replaced Limitations itemize!")

with open(main_tex, "w", encoding="utf-8") as f:
    f.write(text)

res = subprocess.run([str(tectonic_exe), "--print", str(main_tex)], capture_output=True, text=True, cwd=str(paper_dir))
for line in res.stdout.splitlines():
    if "bytes written" in line:
        print(line)
