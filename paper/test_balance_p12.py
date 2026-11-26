from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
main_tex = paper_dir / "main.tex"
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

with open(main_tex, "r", encoding="utf-8") as f:
    text = f.read()

# Add \IEEEtriggeratref{32} right before \begin{thebibliography}
text = text.replace(r"\begin{thebibliography}{38}", r"\IEEEtriggeratref{32}" + "\n" + r"\begin{thebibliography}{38}")

with open(paper_dir / "main.tex", "w", encoding="utf-8") as f:
    f.write(text)

res = subprocess.run([str(tectonic_exe), "--print", str(paper_dir / "main.tex")], capture_output=True, text=True, cwd=str(paper_dir))
for line in res.stdout.splitlines():
    if "bytes written" in line:
        print(line)
