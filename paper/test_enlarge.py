from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

with open(paper_dir / "main.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Let's check \enlargethispage{1\baselineskip} on page 10 or 11
# In LaTeX, \enlargethispage{\baselineskip} allows one extra line on a page!
# If page 11 has \enlargethispage{2\baselineskip}, it easily accommodates the last reference!
# Let's test placing \enlargethispage{2\baselineskip} right after \begin{thebibliography} or \section{Conclusion}

text_test = text.replace(r"\section{Conclusion}", r"\enlargethispage{2\baselineskip}" + "\n" + r"\section{Conclusion}")

with open(paper_dir / "temp_enlarge.tex", "w", encoding="utf-8") as f:
    f.write(text_test)

res = subprocess.run([str(tectonic_exe), "--print", str(paper_dir / "temp_enlarge.tex")], capture_output=True, text=True, cwd=str(paper_dir))

for line in res.stdout.splitlines():
    if "bytes written" in line:
        print(line)
