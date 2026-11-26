from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

with open(paper_dir / "main.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Let's insert probes before Section IV-E, IV-F, Table VI, Table VII
for target in [r"\subsection{Computational Complexity", r"\subsection{Dialectal Phonetic", r"\begin{table}[!t]" + "\n" + r"\centering" + "\n" + r"\caption{Linguistic Error", r"\section{Discussion}"]:
    if target in text:
        text = text.replace(target, f"\\typeout{{PROBE AT {target[:25]}: page=\\thepage}}\n" + target)

with open(paper_dir / "temp_p9_10.tex", "w", encoding="utf-8") as f:
    f.write(text)

res = subprocess.run([str(tectonic_exe), "--print", str(paper_dir / "temp_p9_10.tex")], capture_output=True, text=True, cwd=str(paper_dir))

for line in res.stdout.splitlines():
    if "PROBE AT" in line:
        print(line)
