from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

with open(paper_dir / "main.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Place \enlargethispage{3\baselineskip} inside thebibliography
text_test = text.replace(r"\begin{thebibliography}{38}", r"\begin{thebibliography}{38}" + "\n" + r"\enlargethispage{3\baselineskip}")

with open(paper_dir / "temp_enlarge2.tex", "w", encoding="utf-8") as f:
    f.write(text_test)

res = subprocess.run([str(tectonic_exe), "--print", str(paper_dir / "temp_enlarge2.tex")], capture_output=True, text=True, cwd=str(paper_dir))

for line in res.stdout.splitlines():
    if "bytes written" in line:
        print(line)
