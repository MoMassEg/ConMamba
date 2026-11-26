from pathlib import Path
import subprocess

paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
tectonic_exe = paper_dir / "tools" / "tectonic.exe"

with open(paper_dir / "main.tex", "r", encoding="utf-8") as f:
    text = f.read()

# Add a probe after \end{itemize} in Limitations
probe = r"""
\makeatletter
\typeout{PAGE10 PROBE: page=\thepage, remaining space=\the\dimexpr\pagegoal-\pagetotal\relax}
\makeatother
"""

text_test = text.replace(r"\end{itemize}" + "\n\n" + r"\section{Conclusion}", r"\end{itemize}" + "\n" + probe + "\n" + r"\section{Conclusion}")

with open(paper_dir / "temp_p10.tex", "w", encoding="utf-8") as f:
    f.write(text_test)

res = subprocess.run([str(tectonic_exe), "--print", str(paper_dir / "temp_p10.tex")], capture_output=True, text=True, cwd=str(paper_dir))

for line in res.stdout.splitlines():
    if "PAGE10 PROBE" in line:
        print(line)
